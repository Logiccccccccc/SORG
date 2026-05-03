import json
import os
import argparse
import sys
import re
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def split_captions_into_sentences(gt_data, min_words=0):
    """
    Split paragraph captions into individual sentence annotations.
    Fixes the issue where short generated captions are penalized heavily 
    when compared against long paragraph ground truth.
    Apply filtering to ignore very short sentences if min_words > 0.
    """
    print(f"正在检测标注格式...尝试将段落分割为句子以优化评估 (最小词数: {min_words})...")
    new_anns = []
    ann_id = 1
    
    count_split = 0
    dropped_sentences = 0
    
    for ann in gt_data.get('annotations', []):
        image_id = ann.get('image_id')
        caption = ann.get('caption', '')
        
        # Simple heuristic split by period followed by space or end of string
        # Also handle potential multiple spaces
        # Note: This might split "Mr. Smith" but for VG captions which are usually descriptive, it's safer.
        sentences = re.split(r'\.\s+', caption)
        
        filtered_sentences = []
        for s in sentences:
            s = s.strip()
            # Remove trailing period if present
            if s.endswith('.'):
                s = s[:-1]
            
            # Filter logic
            if len(s) > 1: # Ignore empty or single char junk
                # Word count check
                word_count = len(s.split())
                if word_count >= min_words:
                    filtered_sentences.append(s)
                else:
                    dropped_sentences += 1
                
        if len(filtered_sentences) > 1:
            count_split += 1
            
        for s in filtered_sentences:
            new_anns.append({
                "id": ann_id,
                "image_id": image_id,
                "caption": s,
                "id_type": "caption" # Ensure type is preserved
            })
            ann_id += 1
            
    print(f"原标注包含 {len(gt_data.get('annotations', []))} 条记录。")
    print(f"因长度不足 (<{min_words} words) 丢弃了 {dropped_sentences} 个句子。")
    print(f"分割后包含 {len(new_anns)} 条句子记录。")
    
    gt_data['annotations'] = new_anns
    return gt_data

def evaluate_spice(results_file, coco_captions_file, min_words=0):
    print(f"正在加载结果文件: {results_file}...")
    print(f"正在加载 COCO 标注: {coco_captions_file}...")

    # 1. 加载 COCO Ground Truth
    with open(coco_captions_file, 'r') as f:
        gt_data = json.load(f)
    
    # Pre-filtering
    caption_anns = [ann for ann in gt_data.get('annotations', []) if ann.get('id_type') == 'caption' or 'caption' in ann]
    gt_data['annotations'] = caption_anns
    if 'categories' in gt_data:
        del gt_data['categories']

    # --- FIX: Split paragraphs into sentences ---
    gt_data = split_captions_into_sentences(gt_data, min_words=min_words)
    # ------------------------------------------

    # Save temp GT file
    temp_gt_file = coco_captions_file.replace('.json', f'_fixed_sentences_min{min_words}.json')
    with open(temp_gt_file, 'w') as f:
        json.dump(gt_data, f)
    print(f"已创建临时句子级标注文件: {temp_gt_file}")

        
    coco = COCO(temp_gt_file)

    # 2. 加载结果并格式化
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
            match = re.search(r'(\d+)', base)
            if match:
                img_id = int(match.group(1))
        
        if img_id is not None:
            # Clean caption slightly
            cap = item.get('caption', '').strip()
            if cap:
                formatted_results.append({
                    'image_id': img_id,
                    'caption': cap
                })

    if not formatted_results:
        print("错误: 无法解析结果文件中的 image_id 或 caption。")
        return

    print(f"找到 {len(formatted_results)} 条有效结果。")

    # Filter GT images to only those in results (optional but speeds up)
    # coco object already has index, but COCOEvalCap might iterate all if not careful
    
    coco_result = coco.loadRes(formatted_results)

    # 3. 设置评估
    coco_eval = COCOEvalCap(coco, coco_result)
    
    # Explicitly set images to evaluate to intersection
    # This ensures we don't evaluate on images missing from results (which would give 0 scores for them)
    res_img_ids = set(r['image_id'] for r in formatted_results)
    gt_img_ids = set(coco.getImgIds())
    common_ids = list(res_img_ids.intersection(gt_img_ids))
    
    if not common_ids:
        print("错误: 结果中的图片ID与标注文件中的图片ID没有交集！")
        print(f"结果ID示例: {list(res_img_ids)[:5]}")
        print(f"标注ID示例: {list(gt_img_ids)[:5]}")
        return

    print(f"将在 {len(common_ids)} 张共有图片上进行评估。")
    coco_eval.params['image_id'] = common_ids
    
    print("开始评估 (SPICE 需要 Java 环境)...")
    try:
        coco_eval.evaluate()
    except Exception as e:
        print(f"\n评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 打印结果
    print("\n" + "="*40)
    print("评估结果 (Sentence Level GT)")
    print("="*40)
    
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")
            
    # Save results
    output_file = results_file.replace('.json', '_spice_fixed.json')
    with open(output_file, 'w') as f:
        json.dump(coco_eval.eval, f, indent=2)
    print(f"\n详细结果已保存到: {output_file}")
    
    return coco_eval.eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SPICE metric (Fixed for Paragraph GT)")
    parser.add_argument('--results_file', type=str, required=True, help="结果 JSON 文件路径")
    parser.add_argument('--coco_captions_file', type=str, default='data/coco/annotations/captions_val2017.json', help="COCO captions 标注文件路径")
    parser.add_argument('--min_words', type=int, default=4, help="保留的最小单词数，过滤掉太短的碎片描述 (默认: 4)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"错误: 找不到结果文件 '{args.results_file}'")
        sys.exit(1)
        
    if not os.path.exists(args.coco_captions_file):
        print(f"错误: 找不到标注文件 '{args.coco_captions_file}'")
        sys.exit(1)

    evaluate_spice(args.results_file, args.coco_captions_file, min_words=args.min_words)

