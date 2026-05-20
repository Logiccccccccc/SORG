import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SORG_DIR = os.path.join(PROJECT_ROOT, 'sorg')
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SORG_DIR not in sys.path:
    sys.path.insert(0, SORG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import json
import os
import sys
from tqdm import tqdm
from pycocotools.coco import COCO
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluation_CHAIR import StandardCHAIR

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate CPR and PoC metrics')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the model output JSON file')
    parser.add_argument('--baseline_file', type=str, required=True, help='Path to the baseline output JSON file')
    parser.add_argument('--annotation_file', type=str, default='data/coco/annotations/instances_val2017.json', help='Path to COCO instances file')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save results JSON')
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_gt_objects(coco, img_id, chair_evaluator):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    gt_objects = set()
    for ann in anns:
        if 'category_id' not in ann:
            continue
        cat_id = ann['category_id']
        cat_name = coco.cats[cat_id]['name']
        gt_objects.add(cat_name)
    return gt_objects

def calculate_metrics(model_data, baseline_data, coco, chair_evaluator):
    baseline_map = {item['image_id']: item for item in baseline_data}
    cpr_scores = []
    poc_scores = []
    total_changes = 0
    positive_changes = 0
    total_correct_baseline = 0
    preserved_correct = 0
    for item in tqdm(model_data, desc='Evaluating'):
        img_id = item['image_id']
        if img_id not in baseline_map:
            continue
        baseline_item = baseline_map[img_id]
        model_caption = item['caption']
        baseline_caption = baseline_item['caption']
        model_objects = chair_evaluator._get_mentioned_objects(model_caption)
        baseline_objects = chair_evaluator._get_mentioned_objects(baseline_caption)
        gt_objects = get_gt_objects(coco, img_id, chair_evaluator)
        baseline_correct = baseline_objects.intersection(gt_objects)
        if len(baseline_correct) > 0:
            preserved = baseline_correct.intersection(model_objects)
            cpr = len(preserved) / len(baseline_correct)
            cpr_scores.append(cpr)
            total_correct_baseline += len(baseline_correct)
            preserved_correct += len(preserved)
        deleted_objects = baseline_objects - model_objects
        added_objects = model_objects - baseline_objects
        current_positive_changes = 0
        current_total_changes = len(deleted_objects) + len(added_objects)
        for obj in deleted_objects:
            if obj not in gt_objects:
                current_positive_changes += 1
        for obj in added_objects:
            if obj in gt_objects:
                current_positive_changes += 1
        if current_total_changes > 0:
            poc = current_positive_changes / current_total_changes
            poc_scores.append(poc)
            total_changes += current_total_changes
            positive_changes += current_positive_changes
    avg_cpr = np.mean(cpr_scores) if cpr_scores else 0.0
    avg_poc = np.mean(poc_scores) if poc_scores else 0.0
    global_cpr = preserved_correct / total_correct_baseline if total_correct_baseline > 0 else 0.0
    global_poc = positive_changes / total_changes if total_changes > 0 else 0.0
    return {'CPR_Macro': avg_cpr, 'PoC_Macro': avg_poc, 'CPR_Micro': global_cpr, 'PoC_Micro': global_poc, 'Num_Samples_CPR': len(cpr_scores), 'Num_Samples_PoC': len(poc_scores)}

def main():
    args = parse_args()
    print(f'Loading Model Results: {args.model_file}')
    model_data = load_json(args.model_file)
    print(f'Loading Baseline Results: {args.baseline_file}')
    baseline_data = load_json(args.baseline_file)
    chair_evaluator = StandardCHAIR(args.annotation_file)
    print('Calculating metrics...')
    results = calculate_metrics(model_data, baseline_data, chair_evaluator.coco, chair_evaluator)
    print('\n' + '=' * 40)
    print('Evaluation Results')
    print('=' * 40)
    print(f"Content Preservation Rate (CPR): {results['CPR_Macro']:.4f} (Macro) | {results['CPR_Micro']:.4f} (Micro)")
    print(f"Precision of Change (PoC):       {results['PoC_Macro']:.4f} (Macro) | {results['PoC_Micro']:.4f} (Micro)")
    print('-' * 40)
    print(f"Samples used for CPR: {results['Num_Samples_CPR']}")
    print(f"Samples used for PoC: {results['Num_Samples_PoC']}")
    print('=' * 40)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'Results saved to {args.output_file}')
if __name__ == '__main__':
    main()
