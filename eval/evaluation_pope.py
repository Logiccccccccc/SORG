import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SORG_DIR = os.path.join(PROJECT_ROOT, "sorg")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SORG_DIR not in sys.path:
    sys.path.insert(0, SORG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from caption_generator import MultimodalCaptionGenerator
from config import Config

class PopeEvaluator:
    def __init__(self, model_checkpoint, knowledge_base_path, coco_instances_path, score_weights=None):
        self.generator = MultimodalCaptionGenerator(
            model_checkpoint=model_checkpoint,
            knowledge_base_path=knowledge_base_path,
            score_weights=score_weights,
        )
        print(f"Loading COCO annotations: {coco_instances_path}")
        self.coco = COCO(coco_instances_path)
        self.category_ids = self.coco.getCatIds()
        self.id_to_name = {cat["id"]: cat["name"] for cat in self.coco.loadCats(self.category_ids)}
        self.all_category_names = list(self.id_to_name.values())
        print("Precomputing COCO statistics for POPE sampling strategies...")
        self._precompute_stats()
        print("COCO statistics are ready.")

    def _precompute_stats(self):
        ann_ids = self.coco.getAnnIds()
        anns = self.coco.loadAnns(ann_ids)
        cat_counts = Counter(ann["category_id"] for ann in anns if "category_id" in ann)
        self.sorted_cats_by_freq = [cat_id for cat_id, _ in cat_counts.most_common()]
        self.co_occurrence = defaultdict(Counter)

        for img_id in self.coco.getImgIds():
            anns_in_img = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            cats_in_img = list({ann["category_id"] for ann in anns_in_img if "category_id" in ann})
            for i in range(len(cats_in_img)):
                for j in range(len(cats_in_img)):
                    if i != j:
                        self.co_occurrence[cats_in_img[i]][cats_in_img[j]] += 1

    def _get_negative_samples(self, present_cat_ids, num_neg, strategy):
        neg_pool_ids = [cat_id for cat_id in self.category_ids if cat_id not in present_cat_ids]
        if not neg_pool_ids:
            return []

        if strategy == "random":
            return random.sample(neg_pool_ids, num_neg) if len(neg_pool_ids) > num_neg else neg_pool_ids

        if strategy == "popular":
            popular_negatives = []
            for cat_id in self.sorted_cats_by_freq:
                if cat_id not in present_cat_ids:
                    popular_negatives.append(cat_id)
                    if len(popular_negatives) == num_neg:
                        break
            return popular_negatives

        if strategy == "adversarial":
            candidate_scores = []
            for neg_cat in neg_pool_ids:
                score = sum(self.co_occurrence[present_cat][neg_cat] for present_cat in present_cat_ids)
                candidate_scores.append((neg_cat, score))
            candidate_scores.sort(key=lambda item: item[1], reverse=True)
            return [cat_id for cat_id, _ in candidate_scores[:num_neg]]

        return []

    def generate_pope_questions(self, image_id, num_pos=3, num_neg=3, strategy="random"):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        present_cat_ids = {ann["category_id"] for ann in anns if "category_id" in ann}

        pos_candidates = list(present_cat_ids)
        pos_samples_ids = random.sample(pos_candidates, num_pos) if len(pos_candidates) > num_pos else pos_candidates
        neg_samples_ids = self._get_negative_samples(present_cat_ids, num_neg, strategy)

        questions = []
        for cat_id in pos_samples_ids:
            name = self.id_to_name[cat_id]
            questions.append({"question": f"Is there a {name} in the image?", "ground_truth": "yes", "image_id": image_id})

        for cat_id in neg_samples_ids:
            name = self.id_to_name[cat_id]
            questions.append({"question": f"Is there a {name} in the image?", "ground_truth": "no", "image_id": image_id})

        return questions

    def ask_vlm(self, image, visual_context, question_text, use_context=False):
        model_type = getattr(self.generator, "model_type", "llava")

        if model_type in ["minigpt4", "instructblip"]:
            prompt = self._build_blip_prompt(visual_context, question_text, use_context)
            inputs = self.generator.vlm_processor(images=image, text=prompt, return_tensors="pt").to(self.generator.device)
        elif model_type == "qwen":
            inputs = self._build_qwen_inputs(image, visual_context, question_text, use_context)
        else:
            prompt = self._build_llava_prompt(visual_context, question_text, use_context)
            inputs = self.generator.vlm_processor(text=prompt, images=image, return_tensors="pt").to(self.generator.device)

        with torch.inference_mode():
            output = self.generator.vlm_model.generate(**inputs, max_new_tokens=5, do_sample=False, num_beams=1)

        if model_type == "qwen":
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)]
            response = self.generator.vlm_processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)
        else:
            response = self.generator.vlm_processor.decode(output[0], skip_special_tokens=True)

        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        return "yes" if "yes" in response.lower() else "no"

    def _build_blip_prompt(self, visual_context, question_text, use_context):
        if use_context and visual_context:
            return f'''Question: The following is a list of objects detected or inferred in the image:
{visual_context}

Based on the image and the list above, answer the question.
Question: {question_text}
Answer only with "Yes" or "No".
Answer:'''
        return f'''Question: {question_text}
Answer only with "Yes" or "No".
Answer:'''

    def _build_llava_prompt(self, visual_context, question_text, use_context):
        if use_context and visual_context:
            return f'''USER: <image>
The following is a list of objects detected or inferred in the image:
{visual_context}

Based on the image and the list above, answer the question.
Question: {question_text}
Answer only with "Yes" or "No".

ASSISTANT:'''
        return f'''USER: <image>
{question_text}
Answer only with "Yes" or "No".

ASSISTANT:'''

    def _build_qwen_inputs(self, image, visual_context, question_text, use_context):
        if use_context and visual_context:
            prompt_text = f'''The following is a list of objects detected or inferred in the image:
{visual_context}

Based on the image and the list above, answer the question.
Question: {question_text}
Answer only with "Yes" or "No".'''
        else:
            prompt_text = f'''{question_text}
Answer only with "Yes" or "No".'''

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = self.generator.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.generator.vlm_processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.generator.device)

    def calculate_metrics(self, records):
        tp, fp, tn, fn = 0, 0, 0, 0
        total_yes_pred = 0

        for record in records:
            pred = record["prediction"]
            gt = record["ground_truth"]
            if pred == "yes":
                total_yes_pred += 1
                if gt == "yes":
                    tp += 1
                else:
                    fp += 1
            else:
                if gt == "no":
                    tn += 1
                else:
                    fn += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        yes_percent = total_yes_pred / len(records) if records else 0

        return {
            "Accuracy": accuracy * 100,
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1 Score": f1 * 100,
            "Yes (%)": yes_percent * 100,
        }

    def run_evaluation(self, image_dir, num_images=100, seed=42, output_file=None, ablation_mode="complete"):
        random.seed(seed)
        all_img_ids = self.coco.getImgIds()
        eval_img_ids = random.sample(all_img_ids, num_images) if num_images < len(all_img_ids) else all_img_ids
        strategies = ["random", "popular", "adversarial"]
        final_table_data = {}

        print(f"Running POPE evaluation with ablation mode: {ablation_mode}")
        for strategy in strategies:
            print(f"\nEvaluating strategy: {strategy.upper()} ...")
            records = []
            for img_id in tqdm(eval_img_ids):
                img_info = self.coco.loadImgs(img_id)[0]
                image_path = os.path.join(image_dir, img_info["file_name"])
                if not os.path.exists(image_path):
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                    visual_context = self._build_visual_context_for_image(image, ablation_mode)
                except Exception as exc:
                    print(f"Error processing image {img_id}: {exc}")
                    continue

                questions = self.generate_pope_questions(img_id, strategy=strategy)
                for question in questions:
                    pred = self.ask_vlm(image, visual_context, question["question"], use_context=bool(visual_context))
                    records.append({**question, "prediction": pred})

            final_table_data[strategy] = self.calculate_metrics(records)

        self.print_final_table(final_table_data, ablation_mode)
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_table_data, f, indent=4, ensure_ascii=False)
            print(f"Results saved to: {output_file}")

    def _build_visual_context_for_image(self, image, ablation_mode):
        if ablation_mode == "baseline":
            return ""

        visual_features = self.generator.extract_visual_features(image)
        if ablation_mode == "no_relations":
            visual_features["relations"] = []
        if ablation_mode == "no_clip":
            visual_features["node_features"] = torch.randn_like(visual_features["node_features"])

        if ablation_mode == "no_reasoning":
            n_objects = len(visual_features["objects"])
            reasoning_result = {"existence_probs": torch.zeros(n_objects).to(self.generator.device)}
        else:
            use_uncertainty = ablation_mode != "no_uncertainty"
            reasoning_result = self.generator.perform_causal_reasoning(visual_features, use_uncertainty=use_uncertainty)

        return self.generator.construct_visual_context(visual_features, reasoning_result)

    def print_final_table(self, data, model_name):
        print("\n" + "=" * 85)
        print(f"POPE Evaluation Results - Mode: {model_name}")
        print(f"{'Dataset':<10} {'Strategy':<12} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Yes(%)':<8}")
        print("-" * 85)

        for strategy in ["random", "popular", "adversarial"]:
            if strategy not in data:
                continue
            metrics = data[strategy]
            print(
                f"{'MSCOCO':<10} {strategy.capitalize():<12} "
                f"{metrics['Accuracy']:<8.2f} {metrics['Precision']:<8.2f} "
                f"{metrics['Recall']:<8.2f} {metrics['F1 Score']:<8.2f} {metrics['Yes (%)']:<8.2f}"
            )
        print("-" * 85)

def parse_score_weights(raw_value):
    if raw_value is None:
        return None
    values = tuple(map(float, raw_value.split(",")))
    if len(values) != 3:
        raise ValueError("score_weights must contain exactly three values")
    return values

def auto_select_checkpoint(checkpoint_dir):
    preferred = os.path.join(checkpoint_dir, "gnn_state_5000.pt")
    if os.path.exists(preferred):
        return preferred

    if not os.path.exists(checkpoint_dir):
        return None

    pt_files = [name for name in os.listdir(checkpoint_dir) if name.endswith(".pt")]
    new_style = [name for name in pt_files if name.startswith("new_gnn_epoch_")]
    candidates = new_style if new_style else pt_files
    if not candidates:
        return None

    candidates.sort(key=lambda name: os.path.getmtime(os.path.join(checkpoint_dir, name)), reverse=True)
    return os.path.join(checkpoint_dir, candidates[0])

def main():
    parser = argparse.ArgumentParser(description="Evaluate POPE on MSCOCO.")
    parser.add_argument("--config", type=str, default="configs/infer_llava.yaml", help="Path to the YAML inference config.")
    parser.add_argument("--image_dir", type=str, default=None, help="Image directory.")
    parser.add_argument("--coco_instances", type=str, default=None, help="COCO instances annotation file.")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to evaluate.")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="GNN checkpoint path.")
    parser.add_argument("--knowledge_base", type=str, default=None, help="Knowledge base directory.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the results JSON file.")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="complete",
        choices=["complete", "baseline", "no_reasoning", "no_relations", "no_uncertainty", "no_clip"],
        help="Ablation mode for evaluation.",
    )
    parser.add_argument("--score_weights", type=str, default=None, help='Core object score weights, format: "0.4,0.3,0.3".')
    args = parser.parse_args()

    Config.load_config(args.config)
    Config.create_dirs()

    image_dir = args.image_dir or Config.COCO_VAL_IMG
    coco_instances = args.coco_instances or os.path.join(Config.COCO_ANNOTATIONS, "instances_val2017.json")
    knowledge_base = args.knowledge_base or Config.KNOWLEDGE_BASE_DIR
    output_file = args.output_file or os.path.join(Config.RESULTS_DIR, "pope_results.json")
    score_weights_tuple = parse_score_weights(args.score_weights)

    model_checkpoint = args.model_checkpoint
    if model_checkpoint is None and args.ablation_mode != "baseline":
        model_checkpoint = auto_select_checkpoint(Config.CHECKPOINT_DIR)
        if model_checkpoint:
            print(f"No checkpoint was specified. Auto-selected: {model_checkpoint}")
        else:
            print("Warning: no model checkpoint was specified and none was found. The GNN will use random initialization.")

    evaluator = PopeEvaluator(model_checkpoint, knowledge_base, coco_instances, score_weights=score_weights_tuple)
    evaluator.run_evaluation(
        image_dir=image_dir,
        num_images=args.num_images,
        output_file=output_file,
        ablation_mode=args.ablation_mode,
    )

if __name__ == "__main__":
    main()
