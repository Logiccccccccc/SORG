import argparse
import json
import os
import random
import torch
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from collections import defaultdict, Counter
from caption_generator import MultimodalCaptionGenerator
from config import Config

"""
1. 完整框架 (Ours)
python evaluation_pope.py \
    --image_dir data/coco/val2017 \
    --coco_instances data/coco/annotations/instances_val2017.json \
    --num_images 500 \
    --output_file results/pope_complete.json \
    --ablation_mode complete
    --model_checkpoint checkpoints/new_gnn_epoch_10.pt
Reasoning (无推理)
python evaluation_pope.py \
    --image_dir data/coco/val2017 \
    --coco_instances data/coco/annotations/instances_val2017.json \
    --num_images 500 \
    --output_file results/pope_no_reasoning.json \
    --ablation_mode no_reasoning
    --model_checkpoint checkpoints/new_gnn_epoch_10.pt

Spatial Relations (无空间关系)
python evaluation_pope.py \
    --image_dir data/coco/val2017 \
    --coco_instances data/coco/annotations/instances_val2017.json \
    --num_images 500 \
    --output_file results/pope_no_relations.json \
    --ablation_mode no_relations
    --model_checkpoint checkpoints/new_gnn_epoch_10.pt
Uncertainty (无不确定性)
python evaluation_pope.py \
    --image_dir data/coco/val2017 \
    --coco_instances data/coco/annotations/instances_val2017.json \
    --num_images 500 \
    --output_file results/pope_no_uncertainty.json \
    --ablation_mode no_uncertainty \
    --model_checkpoint checkpoints/new_gnn_epoch_10.pt
    --model_checkpoint checkpoints/new_gnn_epoch_10.pt
CLIP Semantics (无CLIP特征)
python evaluation_pope.py \
    --image_dir data/coco/val2017 \
    --coco_instances data/coco/annotations/instances_val2017.json \
    --num_images 500 \
    --output_file results/pope_no_clip.json \
    --ablation_mode no_clip
    --model_checkpoint checkpoints/new_gnn_epoch_10.pt
Baseline (LLaVA Only)
python evaluation_pope.py \
    --image_dir data/coco/val2017 \
    --coco_instances data/coco/annotations/instances_val2017.json \
    --num_images 500 \
    --output_file results/pope_baseline.json \
    --ablation_mode baseline
    --model_checkpoint checkpoints/new_gnn_epoch_10.pt
"""







class PopeEvaluator:
    def __init__(self, model_checkpoint, knowledge_base_path, coco_instances_path, score_weights=None):
        # ... (保持原有的 __init__ 代码不变) ...
        self.generator = MultimodalCaptionGenerator(
            model_checkpoint=model_checkpoint,
            knowledge_base_path=knowledge_base_path,
            score_weights=score_weights
        )
        
        print(f"Loading COCO annotations: {coco_instances_path}")
        self.coco = COCO(coco_instances_path)
        self.category_ids = self.coco.getCatIds()
        self.id_to_name = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.category_ids)}
        self.all_category_names = list(self.id_to_name.values())

        print("Precomputing COCO stats for sampling strategies...")
        self._precompute_stats()
        print("Stats ready.")

    def _precompute_stats(self):
        # ... (保持原有代码不变) ...
        ann_ids = self.coco.getAnnIds()
        anns = self.coco.loadAnns(ann_ids)
        cat_counts = Counter([ann['category_id'] for ann in anns])
        self.sorted_cats_by_freq = [cat_id for cat_id, _ in cat_counts.most_common()]

        self.co_occurrence = defaultdict(Counter)
        img_ids = self.coco.getImgIds()
        for img_id in img_ids:
            anns_in_img = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            cats_in_img = list(set([ann['category_id'] for ann in anns_in_img]))
            for i in range(len(cats_in_img)):
                for j in range(len(cats_in_img)):
                    if i != j:
                        self.co_occurrence[cats_in_img[i]][cats_in_img[j]] += 1

    def _get_negative_samples(self, present_cat_ids, num_neg, strategy):
        # ... (保持原有代码不变) ...
        neg_pool_ids = [c for c in self.category_ids if c not in present_cat_ids]
        if not neg_pool_ids: return []

        if strategy == 'random':
            if len(neg_pool_ids) > num_neg:
                return random.sample(neg_pool_ids, num_neg)
            return neg_pool_ids

        elif strategy == 'popular':
            popular_negatives = []
            for cat_id in self.sorted_cats_by_freq:
                if cat_id not in present_cat_ids:
                    popular_negatives.append(cat_id)
                    if len(popular_negatives) == num_neg: break
            return popular_negatives

        elif strategy == 'adversarial':
            candidate_scores = []
            for neg_cat in neg_pool_ids:
                score = 0
                for present_cat in present_cat_ids:
                    score += self.co_occurrence[present_cat][neg_cat]
                candidate_scores.append((neg_cat, score))
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            return [x[0] for x in candidate_scores[:num_neg]]

        return []

    def generate_pope_questions(self, image_id, num_pos=3, num_neg=3, strategy='random'):
        # ... (保持原有代码不变) ...
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        present_cat_ids = set([ann['category_id'] for ann in anns])
        
        pos_candidates = list(present_cat_ids)
        pos_samples_ids = random.sample(pos_candidates, num_pos) if len(pos_candidates) > num_pos else pos_candidates
            
        neg_samples_ids = self._get_negative_samples(present_cat_ids, num_neg, strategy)
            
        questions = []
        for cat_id in pos_samples_ids:
            name = self.id_to_name[cat_id]
            questions.append({'question': f"Is there a {name} in the image?", 'ground_truth': 'yes', 'image_id': image_id})
            
        for cat_id in neg_samples_ids:
            name = self.id_to_name[cat_id]
            questions.append({'question': f"Is there a {name} in the image?", 'ground_truth': 'no', 'image_id': image_id})
            
        return questions

    def ask_vlm(self, image, visual_context, question_text, use_context=False):
        model_type = getattr(self.generator, 'model_type', 'llava')
        
        # === MiniGPT-4 / InstructBLIP Logic ===
        if model_type in ['minigpt4', 'instructblip']:
            if use_context and visual_context:
                # InstructBLIP is better at pure instruction, but maintaining consistent QA format helps
                prompt = f"""Question: The following is a list of objects detected or inferred in the image:
{visual_context}

Based on the image and the list above, answer the question.
Question: {question_text}
Answer only with "Yes" or "No".
Answer:"""
            else:
                prompt = f"""Question: {question_text}
Answer only with "Yes" or "No".
Answer:"""
            
            inputs = self.generator.vlm_processor(images=image, text=prompt, return_tensors="pt").to(self.generator.device)

        # === Qwen Logic ===
        elif model_type == 'qwen':
            if use_context and visual_context:
                prompt_text = f"""The following is a list of objects detected or inferred in the image:
{visual_context}

Based on the image and the list above, answer the question.
Question: {question_text}
Answer only with "Yes" or "No"."""
            else:
                prompt_text = f"""{question_text}
Answer only with "Yes" or "No"."""

            # Messages format for apply_chat_template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            # Generate input text with special tokens
            text = self.generator.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.generator.vlm_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.generator.device)

        # === LLaVA Logic ===
        else:
            if use_context and visual_context:
                prompt = f"""USER: <image>
The following is a list of objects detected or inferred in the image:
{visual_context}

Based on the image and the list above, answer the question.
Question: {question_text}
Answer only with "Yes" or "No".

ASSISTANT:"""
            else:
                prompt = f"""USER: <image>
{question_text}
Answer only with "Yes" or "No".

ASSISTANT:"""

            inputs = self.generator.vlm_processor(text=prompt, images=image, return_tensors="pt").to(self.generator.device)
        
        with torch.inference_mode():
            output = self.generator.vlm_model.generate(
                **inputs, max_new_tokens=5, do_sample=False, num_beams=1
            )
        
        # FIX: For Qwen (and other decoder-only models returning full sequence), we must trim the input tokens
        if model_type == 'qwen':
             generated_ids = output
             generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
             response = self.generator.vlm_processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)
        else:
            response = self.generator.vlm_processor.decode(output[0], skip_special_tokens=True)

        # Cleanup
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        if "Answer:" in response:
            try:
                # Sometimes MiniGPT-4 puts answer after Answer:
                response = response.split("Answer:")[-1].strip()
            except:
                pass
        
        return "yes" if "yes" in response.lower() else "no"

    def calculate_metrics(self, records):
        # ... (保持原有代码不变) ...
        tp, fp, tn, fn = 0, 0, 0, 0
        total_yes_pred = 0
        
        for r in records:
            pred = r['prediction']
            gt = r['ground_truth']
            
            if pred == 'yes':
                total_yes_pred += 1
                if gt == 'yes': tp += 1
                else: fp += 1
            else:
                if gt == 'no': tn += 1
                else: fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        yes_percent = total_yes_pred / len(records) if len(records) > 0 else 0
        
        return {
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1 * 100,
            'Yes (%)': yes_percent * 100
        }

    # ================= 修改开始：增加了 output_file 参数 =================
    def run_evaluation(self, image_dir, num_images=100, seed=42, output_file=None, ablation_mode='complete'):
        random.seed(seed)
        all_img_ids = self.coco.getImgIds()
        eval_img_ids = random.sample(all_img_ids, num_images) if num_images < len(all_img_ids) else all_img_ids
        
        strategies = ['random', 'popular', 'adversarial']
        final_table_data = {} 

        print(f"Running POPE evaluation with ablation mode: {ablation_mode}")

        for strategy in strategies:
            print(f"\nEvaluating Strategy: {strategy.upper()} ...")
            
            records = []
            
            for img_id in tqdm(eval_img_ids):
                img_info = self.coco.loadImgs(img_id)[0]
                image_path = os.path.join(image_dir, img_info['file_name'])
                if not os.path.exists(image_path): continue
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    
                    # === 消融实验逻辑 (复刻自 inference.py) ===
                    visual_context = ""
                    
                    if ablation_mode == 'baseline':
                        # Baseline: 不使用任何视觉增强
                        visual_context = ""
                    else:
                        # 1. 提取特征
                        visual_features = self.generator.extract_visual_features(image)
                        
                        # [消融] no_relations
                        if ablation_mode == 'no_relations':
                            visual_features['relations'] = []
                            
                        # [消融] no_clip
                        if ablation_mode == 'no_clip':
                            visual_features['node_features'] = torch.randn_like(visual_features['node_features'])
                        
                        # 2. 推理
                        if ablation_mode == 'no_reasoning':
                            # 跳过推理
                            n_objects = len(visual_features['objects'])
                            reasoning_result = {
                                'existence_probs': torch.zeros(n_objects).to(self.generator.device)
                            }
                        else:
                            # 正常推理 (可控不确定性)
                            use_uncertainty = (ablation_mode != 'no_uncertainty')
                            reasoning_result = self.generator.perform_causal_reasoning(
                                visual_features, 
                                use_uncertainty=use_uncertainty
                            )
                        
                        # 3. 构建上下文
                        visual_context = self.generator.construct_visual_context(visual_features, reasoning_result)

                except Exception as e:
                    print(f"Error processing image {img_id}: {e}")
                    continue

                questions = self.generate_pope_questions(img_id, strategy=strategy)
                
                for q in questions:
                    # 如果 visual_context 为空，ask_llava 会自动使用无 context 的 prompt
                    # use_context=True 意为“尝试使用 context”，如果 context 为空字符串，效果等同于 False
                    pred = self.ask_vlm(image, visual_context, q['question'], use_context=bool(visual_context))
                    records.append({**q, 'prediction': pred})

            final_table_data[strategy] = self.calculate_metrics(records)

        # 1. Print to Terminal
        self.print_final_table(final_table_data, ablation_mode)

        # 2. Save to JSON File
        if output_file:
            print(f"\nSaving results to {output_file} ...")
            try:
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_table_data, f, indent=4, ensure_ascii=False)
                print("Save successful.")
            except Exception as e:
                print(f"Error saving JSON: {e}")

    def print_final_table(self, data, model_name):
        print("\n" + "="*85)
        print(f"POPE Evaluation Results - Mode: {model_name}")
        print(f"{'Dataset':<10} {'Strategy':<12} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Yes(%)':<8}")
        print("-" * 85)
        
        dataset_name = "MSCOCO"
        
        for strategy in ['random', 'popular', 'adversarial']:
            if strategy not in data: continue
            metrics = data[strategy]
            
            strat_str = strategy.capitalize()
            
            print(f"{dataset_name:<10} {strat_str:<12} "
                  f"{metrics['Accuracy']:<8.2f} {metrics['Precision']:<8.2f} "
                  f"{metrics['Recall']:<8.2f} {metrics['F1 Score']:<8.2f} {metrics['Yes (%)']:<8.2f}")
            
        print("-" * 85)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/coco/val2017')
    parser.add_argument('--coco_instances', type=str, default='data/coco/annotations/instances_val2017.json')
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--knowledge_base', type=str, default='data/knowledge_base')
    
    # ================= 新增参数 =================
    parser.add_argument('--output_file', type=str, default='pope_results.json', help='Path to save the results JSON file')
    parser.add_argument('--ablation_mode', type=str, default='complete', 
                        choices=['complete', 'baseline', 'no_reasoning', 'no_relations', 'no_uncertainty', 'no_clip'],
                        help='Ablation mode for evaluation')
    parser.add_argument('--score_weights', type=str, default=None,
                        help='核心对象评分权重，格式: "0.4,0.3,0.3"')
    # ===========================================
    
    args = parser.parse_args()
    Config.create_dirs()

    # 解析 score_weights
    score_weights_tuple = None
    if args.score_weights:
        try:
            score_weights_tuple = tuple(map(float, args.score_weights.split(',')))
            if len(score_weights_tuple) != 3:
                raise ValueError("Must provide 3 weights")
        except Exception as e:
            print(f"Invalid score weights: {e}")
            return

    # === 自动检查 checkpoint ===
    if args.model_checkpoint is None and args.ablation_mode != 'baseline':
        # 尝试寻找默认的 checkpoint
        potential_checkpoints = [
            'checkpoints/new_gnn_epoch_10.pt',
            'checkpoint_CLIP/new_gnn_epoch_3.pt',
            'results/gnn_state_5000.pth'
        ]
        for ckpt in potential_checkpoints:
            if os.path.exists(ckpt):
                print(f"Warning: No checkpoint specified. Auto-loading found checkpoint: {ckpt}")
                args.model_checkpoint = ckpt
                break
        
        if args.model_checkpoint is None:
            print("\n" + "!"*80)
            print("CRITICAL WARNING: No model checkpoint specified and none found!")
            print("The GNN will use RANDOM initialization. Results will be meaningless.")
            print("Please specify --model_checkpoint path/to/model.pt")
            print("!"*80 + "\n")

    evaluator = PopeEvaluator(args.model_checkpoint, args.knowledge_base, args.coco_instances, score_weights=score_weights_tuple)
    
    evaluator.run_evaluation(
        image_dir=args.image_dir, 
        num_images=args.num_images,
        output_file=args.output_file,
        ablation_mode=args.ablation_mode
    )

if __name__ == "__main__":
    main()