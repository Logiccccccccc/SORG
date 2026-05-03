import argparse
import json
import os
import sys
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np

# Add current directory to path to import evaluation_CHAIR

"""
cd vision-caption-correction
python evaluate_cpr_poc.py \
    --model_file results/full_results.json \
    --baseline_file results/test_5000_baseline.json \
    --annotation_file data/coco/annotations/instances_val2017.json
--model_file: 你的模型推理结果 JSON 文件
--baseline_file: Baseline 模型的推理结果 JSON 文件。

正确内容保留率 (Content Preservation Rate, CPR)：在 Baseline 中原本正确的对象，有多少比例在你的模型输出中依然存在
修改精准度 (Precision of Change, PoC)：模型做的所有“实质性修改”（删除一个对象 或 新增一个对象）中，有多少是正向收益的
"""
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation_CHAIR import StandardCHAIR

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate CPR and PoC metrics")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the model output JSON file")
    parser.add_argument("--baseline_file", type=str, required=True, help="Path to the baseline output JSON file")
    parser.add_argument("--annotation_file", type=str, default="data/coco/annotations/instances_val2017.json", help="Path to COCO instances file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON")
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_gt_objects(coco, img_id, chair_evaluator):
    """Get Ground Truth objects for an image using CHAIR evaluator's mapping"""
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    gt_objects = set()
    for ann in anns:
        if 'category_id' not in ann:
            continue
        cat_id = ann['category_id']
        cat_name = coco.cats[cat_id]['name']
        # Use the same normalization as CHAIR
        # CHAIR maps words to categories. Here we have categories.
        # We need to ensure the format matches what _get_mentioned_objects returns.
        # _get_mentioned_objects returns COCO category names.
        gt_objects.add(cat_name)
    return gt_objects

def calculate_metrics(model_data, baseline_data, coco, chair_evaluator):
    # Create a map for baseline data for quick lookup
    baseline_map = {item['image_id']: item for item in baseline_data}
    
    # Metrics accumulators
    cpr_scores = []
    poc_scores = []
    
    total_changes = 0
    positive_changes = 0
    
    total_correct_baseline = 0
    preserved_correct = 0
    
    for item in tqdm(model_data, desc="Evaluating"):
        img_id = item['image_id']
        
        if img_id not in baseline_map:
            continue
            
        baseline_item = baseline_map[img_id]
        
        # Get captions
        model_caption = item['caption']
        baseline_caption = baseline_item['caption']
        
        # Extract objects
        model_objects = chair_evaluator._get_mentioned_objects(model_caption)
        baseline_objects = chair_evaluator._get_mentioned_objects(baseline_caption)
        
        # Get GT objects
        gt_objects = get_gt_objects(coco, img_id, chair_evaluator)
        
        # --- Calculate CPR (Content Preservation Rate) ---
        # Correct objects in Baseline
        baseline_correct = baseline_objects.intersection(gt_objects)
        
        if len(baseline_correct) > 0:
            # How many are still in Model output
            preserved = baseline_correct.intersection(model_objects)
            cpr = len(preserved) / len(baseline_correct)
            cpr_scores.append(cpr)
            
            total_correct_baseline += len(baseline_correct)
            preserved_correct += len(preserved)
        
        # --- Calculate PoC (Precision of Change) ---
        # Deletions: In Baseline but not in Model
        deleted_objects = baseline_objects - model_objects
        # Additions: In Model but not in Baseline
        added_objects = model_objects - baseline_objects
        
        current_positive_changes = 0
        current_total_changes = len(deleted_objects) + len(added_objects)
        
        # Check Deletions (Positive if it was NOT in GT -> Hallucination removal)
        for obj in deleted_objects:
            if obj not in gt_objects:
                current_positive_changes += 1
                
        # Check Additions (Positive if it IS in GT -> Correction)
        for obj in added_objects:
            if obj in gt_objects:
                current_positive_changes += 1
        
        if current_total_changes > 0:
            poc = current_positive_changes / current_total_changes
            poc_scores.append(poc)
            
            total_changes += current_total_changes
            positive_changes += current_positive_changes

    # Aggregate results
    avg_cpr = np.mean(cpr_scores) if cpr_scores else 0.0
    avg_poc = np.mean(poc_scores) if poc_scores else 0.0
    
    # Global calculation (Micro-average)
    global_cpr = preserved_correct / total_correct_baseline if total_correct_baseline > 0 else 0.0
    global_poc = positive_changes / total_changes if total_changes > 0 else 0.0
    
    return {
        "CPR_Macro": avg_cpr,
        "PoC_Macro": avg_poc,
        "CPR_Micro": global_cpr,
        "PoC_Micro": global_poc,
        "Num_Samples_CPR": len(cpr_scores),
        "Num_Samples_PoC": len(poc_scores)
    }

def main():
    args = parse_args()
    
    print(f"Loading Model Results: {args.model_file}")
    model_data = load_json(args.model_file)
    
    print(f"Loading Baseline Results: {args.baseline_file}")
    baseline_data = load_json(args.baseline_file)
    
    # Initialize CHAIR evaluator (it handles COCO loading and synonyms)
    chair_evaluator = StandardCHAIR(args.annotation_file)
    
    print("Calculating metrics...")
    results = calculate_metrics(model_data, baseline_data, chair_evaluator.coco, chair_evaluator)
    
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Content Preservation Rate (CPR): {results['CPR_Macro']:.4f} (Macro) | {results['CPR_Micro']:.4f} (Micro)")
    print(f"Precision of Change (PoC):       {results['PoC_Macro']:.4f} (Macro) | {results['PoC_Micro']:.4f} (Micro)")
    print("-" * 40)
    print(f"Samples used for CPR: {results['Num_Samples_CPR']}")
    print(f"Samples used for PoC: {results['Num_Samples_PoC']}")
    print("="*40)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
