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
import json
import os
import argparse
import sys
import re
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def split_captions_into_sentences(gt_data, min_words=0):
    print(f'Checking annotation format and splitting paragraphs into sentences for better evaluation (minimum words: {min_words})...')
    new_anns = []
    ann_id = 1
    count_split = 0
    dropped_sentences = 0
    for ann in gt_data.get('annotations', []):
        image_id = ann.get('image_id')
        caption = ann.get('caption', '')
        sentences = re.split('\\.\\s+', caption)
        filtered_sentences = []
        for s in sentences:
            s = s.strip()
            if s.endswith('.'):
                s = s[:-1]
            if len(s) > 1:
                word_count = len(s.split())
                if word_count >= min_words:
                    filtered_sentences.append(s)
                else:
                    dropped_sentences += 1
        if len(filtered_sentences) > 1:
            count_split += 1
        for s in filtered_sentences:
            new_anns.append({'id': ann_id, 'image_id': image_id, 'caption': s, 'id_type': 'caption'})
            ann_id += 1
    print(f"Original annotations contain {len(gt_data.get('annotations', []))} records.")
    print(f'Dropped {dropped_sentences} sentences shorter than {min_words} words.')
    print(f'After splitting, there are {len(new_anns)} sentence records.')
    gt_data['annotations'] = new_anns
    return gt_data

def evaluate_spice(results_file, coco_captions_file, min_words=0):
    print(f'Loading results file: {results_file}...')
    print(f'Loading COCO annotations: {coco_captions_file}...')
    with open(coco_captions_file, 'r') as f:
        gt_data = json.load(f)
    caption_anns = [ann for ann in gt_data.get('annotations', []) if ann.get('id_type') == 'caption' or 'caption' in ann]
    gt_data['annotations'] = caption_anns
    if 'categories' in gt_data:
        del gt_data['categories']
    gt_data = split_captions_into_sentences(gt_data, min_words=min_words)
    temp_gt_file = coco_captions_file.replace('.json', f'_fixed_sentences_min{min_words}.json')
    with open(temp_gt_file, 'w') as f:
        json.dump(gt_data, f)
    print(f'Created temporary sentence-level annotation file: {temp_gt_file}')
    coco = COCO(temp_gt_file)
    with open(results_file, 'r') as f:
        results = json.load(f)
    formatted_results = []
    if isinstance(results, dict):
        results = [results]
    for item in results:
        img_id = None
        if 'image_id' in item:
            try:
                img_id = int(item['image_id'])
            except:
                pass
        if img_id is None and 'image_path' in item:
            base = os.path.basename(item['image_path'])
            import re
            match = re.search('(\\d+)', base)
            if match:
                img_id = int(match.group(1))
        if img_id is not None:
            cap = item.get('caption', '').strip()
            if cap:
                formatted_results.append({'image_id': img_id, 'caption': cap})
    if not formatted_results:
        print('Error: failed to parse image_id or caption from the results file.')
        return
    print(f'Found {len(formatted_results)} valid results.')
    coco_result = coco.loadRes(formatted_results)
    coco_eval = COCOEvalCap(coco, coco_result)
    res_img_ids = set((r['image_id'] for r in formatted_results))
    gt_img_ids = set(coco.getImgIds())
    common_ids = list(res_img_ids.intersection(gt_img_ids))
    if not common_ids:
        print('Error: there is no overlap between result image IDs and annotation image IDs.')
        print(f'Example result IDs: {list(res_img_ids)[:5]}')
        print(f'Example annotation IDs: {list(gt_img_ids)[:5]}')
        return
    print(f'Evaluating on {len(common_ids)} overlapping images.')
    coco_eval.params['image_id'] = common_ids
    print('Starting evaluation. SPICE requires a Java environment...')
    try:
        coco_eval.evaluate()
    except Exception as e:
        print(f'\nAn error occurred during evaluation: {e}')
        import traceback
        traceback.print_exc()
        return
    print('\n' + '=' * 40)
    print('Evaluation Results (Sentence-Level GT)')
    print('=' * 40)
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.4f}')
    output_file = results_file.replace('.json', '_spice_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(coco_eval.eval, f, indent=2)
    print(f'\nDetailed results saved to: {output_file}')
    return coco_eval.eval
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SPICE metric (Fixed for Paragraph GT)')
    parser.add_argument('--results_file', type=str, required=True, help='Path to the result JSON file.')
    parser.add_argument('--coco_captions_file', type=str, default='data/coco/annotations/captions_val2017.json', help='Path to the COCO captions annotation file.')
    parser.add_argument('--min_words', type=int, default=4, help='Minimum words to keep; filters out short fragments. Default: 4.')
    args = parser.parse_args()
    if not os.path.exists(args.results_file):
        print(f"Error: result file not found '{args.results_file}'")
        sys.exit(1)
    if not os.path.exists(args.coco_captions_file):
        print(f"Error: annotation file not found '{args.coco_captions_file}'")
        sys.exit(1)
    evaluate_spice(args.results_file, args.coco_captions_file, min_words=args.min_words)
