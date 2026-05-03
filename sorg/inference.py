"""推理模块 - 支持单张和批量图像描述生成、

qwen
python inference.py \
    --mode batch \
    --model_type qwen \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/result_qwen_ours.json \
    --max_samples 20


使用instructblip模型
python inference.py \
    --mode batch \
    --model_type instructblip \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/result_instructblip_ours.json \
    --max_samples 20



1. 完整框架 (Ours)
python inference.py \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/ablation_complete.json \
    --ablation_mode complete

2. Baseline (LLaVA Only)
python inference.py \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/ablation_baseline.json \
    --ablation_mode baseline

3. w/o Reasoning (无推理/仅检测)
python inference.py \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/ablation_no_reasoning.json \
    --ablation_mode no_reasoning

4. w/o Spatial Relations (无空间关系)
python inference.py \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/ablation_no_relations.json \
    --ablation_mode no_relations

5. w/o Uncertainty (无不确定性估计)
python inference.py \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/ablation_no_uncertainty.json \
    --ablation_mode no_uncertainty
CLIP 消融实验：
python inference.py \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/ablation_no_clip.json \
    --ablation_mode no_clip
"""
import torch
import os
import json
import argparse
import re
from tqdm import tqdm
from PIL import Image

from config import Config
from caption_generator import MultimodalCaptionGenerator


def extract_image_id(image_path):
    """从图像路径提取 image_id"""
    filename = os.path.basename(image_path)
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


def run_single_inference(args):
    """单张图像推理"""
    print("="*60)
    print("初始化生成器...")
    
    # 映射旧参数 baseline 到新的 ablation_mode
    if args.baseline and args.ablation_mode == 'complete':
        args.ablation_mode = 'baseline'
        
    print(f"【模式】: {args.ablation_mode}")
    if args.ablation_mode == 'baseline':
        print("  - 仅使用 LLaVA，跳过 GNN 和知识库")
    elif args.ablation_mode == 'no_reasoning':
        print("  - 仅使用视觉检测，跳过 GNN 推理")
    elif args.ablation_mode == 'no_relations':
        print("  - 禁用空间关系 (无边图)")
    elif args.ablation_mode == 'no_uncertainty':
        print("  - 禁用不确定性估计 (单次前向)")
    elif args.ablation_mode == 'no_clip':
        print("  - 禁用 CLIP 特征 (使用随机特征)")
    elif args.ablation_mode.startswith('selector_'):
        print(f"  - 核心对象选择消融: {args.ablation_mode}")
    else:
        print("  - 完整模式: 使用 GNN、空间关系、CLIP特征和不确定性增强")
    print("="*60)
    
    generator = MultimodalCaptionGenerator(
        model_checkpoint=args.model_checkpoint,
        knowledge_base_path=args.knowledge_base,
        score_weights=args.score_weights_tuple
    )
    
    print("\n" + "="*60)
    print(f"处理图像: {args.image_path}")
    print("="*60)
    
    try:
        result = generator.generate(args.image_path, ablation_mode=args.ablation_mode)
        
        # 打印结果
        print("\n" + "="*60)
        print("生成结果")
        print("="*60)
        print(f"\n【生成的描述】")
        print(result['caption'])
        
        print(f"\n【视觉上下文】")
        print(result['visual_context'])
        
        print(f"\n【检测对象】({len(result['detected_objects'])} 个)")
        for obj in result['detected_objects'][:10]:  # 只显示前10个
            core_tag = " [核心]" if obj['is_core'] else ""
            print(f"  - {obj['label']}: {obj['score']:.2f}{core_tag}")
        
        if result['inferred_objects']:
            print(f"\n【推断对象】({len(result['inferred_objects'])} 个)")
            for obj in result['inferred_objects'][:5]:  # 只显示前5个
                print(f"  - {obj['label']}: {obj['existence_prob']:.2f}")
        
        # 保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 结果已保存到: {args.output_file}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 错误: 处理失败")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_batch_inference(args):
    """批量推理"""
    print("="*60)
    print("批量推理模式")
    
    # 映射旧参数 baseline 到新的 ablation_mode
    if args.baseline and args.ablation_mode == 'complete':
        args.ablation_mode = 'baseline'

    print(f"【模式】: {args.ablation_mode}")
    if args.ablation_mode == 'baseline':
        print("  - 仅使用 LLaVA，跳过 GNN 和知识库")
    elif args.ablation_mode == 'no_reasoning':
        print("  - 仅使用视觉检测，跳过 GNN 推理")
    elif args.ablation_mode == 'no_relations':
        print("  - 禁用空间关系 (无边图)")
    elif args.ablation_mode == 'no_uncertainty':
        print("  - 禁用不确定性估计 (单次前向)")
    elif args.ablation_mode == 'no_clip':
        print("  - 禁用 CLIP 特征 (使用随机特征)")
    elif args.ablation_mode.startswith('selector_'):
        print(f"  - 核心对象选择消融: {args.ablation_mode}")
    else:
        print("  - 完整模式: 使用 GNN、空间关系、CLIP特征和不确定性增强")
    print("="*60)
    
    # 加载图像列表
    if args.annotation_file:
        print(f"从标注文件加载图像列表: {args.annotation_file}")
        with open(args.annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        image_paths = []
        missing_count = 0
        max_needed = args.max_samples if args.max_samples and args.max_samples > 0 else None

        # 关键：不要直接截断 images[:max_samples]，否则遇到缺失文件会导致数量变少。
        # 这里顺序遍历整个列表，跳过不存在图片，并继续往后补齐直到凑满 max_samples。
        for img_info in coco_data.get('images', []):
            if max_needed is not None and len(image_paths) >= max_needed:
                break
            file_name = img_info.get('file_name')
            if not file_name:
                continue
            img_path = os.path.join(args.image_dir, file_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                missing_count += 1

        if max_needed is None:
            print(f"找到 {len(image_paths)} 张有效图像（未限制 max_samples）")
        else:
            print(f"目标 {max_needed} 张，找到 {len(image_paths)} 张有效图像（跳过缺失 {missing_count} 张）")
    
    elif args.image_list:
        print(f"从文件列表加载图像: {args.image_list}")
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        # 验证路径
        image_paths = [p for p in image_paths if os.path.exists(p)]
        print(f"找到 {len(image_paths)} 张有效图像")
    
    elif args.image_dir:
        print(f"从目录加载图像: {args.image_dir}")
        image_paths = []
        for fname in os.listdir(args.image_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(args.image_dir, fname))
        
        if args.max_samples > 0:
            image_paths = image_paths[:args.max_samples]
        
        print(f"找到 {len(image_paths)} 张图像")
    
    else:
        print("❌ 错误: 批量模式需要指定 --annotation_file、--image_list 或 --image_dir")
        return None
    
    if not image_paths:
        print("❌ 错误: 没有找到任何图像")
        return None
    
    # 初始化生成器
    print("\n初始化生成器...")
    generator = MultimodalCaptionGenerator(
        model_checkpoint=args.model_checkpoint,
        knowledge_base_path=args.knowledge_base,
        score_weights=args.score_weights_tuple
    )
    
    # 批量处理
    print(f"\n开始处理 {len(image_paths)} 张图像...")
    results = []
    success_count = 0
    failed_count = 0
    
    for img_path in tqdm(image_paths, desc="生成描述"):
        try:
            result = generator.generate(img_path, ablation_mode=args.ablation_mode)
            
            # 确保包含必要字段
            if 'image_id' not in result:
                result['image_id'] = extract_image_id(img_path)
            
            # --- 核心修改：确保这里的字段与之前的格式一致 ---
            # 之前返回的是: image_id, image_path, image_file, caption, visual_context, detected_objects, inferred_objects
            
            # 展平 detected_objects (从 visual_features.objects)
            if 'detected_objects' not in result and 'visual_features' in result and 'objects' in result['visual_features']:
                 result['detected_objects'] = result['visual_features']['objects']
                 
            # 展平 inferred_objects (需要从 reasoning_result 解析，或者 generator 应该直接返回更好的格式)
            if 'inferred_objects' not in result:
                 result['inferred_objects'] = []

            if 'caption' not in result or not result['caption']:
                result['caption'] = ""
            
            # 构造新的 OrderedDict 以保证字段顺序
            from collections import OrderedDict
            final_result = OrderedDict()
            final_result['image_id'] = result['image_id']
            final_result['image_path'] = img_path
            final_result['image_file'] = os.path.basename(img_path)
            final_result['caption'] = result['caption']
            final_result['visual_context'] = result.get('visual_context', '')
            final_result['detected_objects'] = result.get('detected_objects', [])
            final_result['inferred_objects'] = result.get('inferred_objects', [])
            
            # 用户要求移除 visual_features 和 reasoning_result
            # 只保留上述核心字段，或者如果有 'error' 字段则保留
            if 'error' in result:
                final_result['error'] = result['error']
            
            results.append(final_result)
            success_count += 1
            
        except Exception as e:
            print(f"\n✗ 处理失败: {img_path}")
            print(f"  错误: {str(e)}")
            
            results.append({
                'image_id': extract_image_id(img_path),
                'image_path': img_path,
                'image_file': os.path.basename(img_path),
                'caption': "",
                'error': str(e)
            })
            failed_count += 1
    
    # 保存结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("批量推理完成")
        print("="*60)
        print(f"总数: {len(image_paths)}")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}")
        print(f"\n✓ 结果已保存到: {args.output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='图像描述生成推理')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'batch'],
                        help='推理模式: single(单张) 或 batch(批量)')
    
    # 单张推理参数
    parser.add_argument('--image_path', type=str,
                        help='单张图像路径（single模式）')
    
    # 批量推理参数
    parser.add_argument('--annotation_file', type=str,
                        help='COCO标注文件（batch模式）')
    parser.add_argument('--image_dir', type=str,
                        help='图像目录（batch模式）')
    parser.add_argument('--image_list', type=str,
                        help='图像路径列表文件（batch模式）')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='最大处理图像数量')
    
    # 模型参数
    parser.add_argument('--model_checkpoint', type=str,
                        default=None,
                        help='GNN模型检查点路径')
    parser.add_argument('--knowledge_base', type=str,
                        default=None,
                        help='知识库路径')
    
    # 输出参数
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出JSON文件路径')
    
    # 其他参数
    parser.add_argument('--baseline', action='store_true',
                        help='[已弃用，请使用 --ablation_mode baseline] 使用Baseline模式')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['llava', 'minigpt4', 'instructblip', 'qwen'],
                        help='覆盖 config.py 中的 MODEL_TYPE 设置')
    parser.add_argument('--ablation_mode', type=str, default='complete', 
                        choices=['complete', 'baseline', 'no_reasoning', 'no_relations', 'no_uncertainty', 'no_clip',
                                 'selector_confidence_only', 'selector_no_relations', 'selector_random'],
                        help='消融实验模式: complete, baseline, no_reasoning, no_relations, no_uncertainty, no_clip, selector_confidence_only, selector_no_relations, selector_random')
    
    parser.add_argument('--score_weights', type=str, default=None,
                        help='核心对象评分权重，格式: "0.4,0.3,0.3"')

    args = parser.parse_args()
    
    # 允许通过命令行覆盖 MODEL_TYPE
    if args.model_type:
        Config.MODEL_TYPE = args.model_type
        print(f"命令强制指定模型类型: {Config.MODEL_TYPE}")

    # 解析 score_weights
    if args.score_weights:
        try:
            args.score_weights_tuple = tuple(map(float, args.score_weights.split(',')))
            if len(args.score_weights_tuple) != 3:
                raise ValueError("Must provide 3 weights")
        except Exception as e:
            print(f"Invalid score weights: {e}")
            return
    else:
        args.score_weights_tuple = None

    # 设置默认值（仅在未指定时自动寻找）
    if not args.model_checkpoint:
        ckpt_dir = Config.CHECKPOINT_DIR

        # 1) 优先使用训练脚本默认保存的综合权重（包含 gnn + feature_builder）
        preferred = os.path.join(ckpt_dir, 'gnn_state_5000.pt')
        if os.path.exists(preferred):
            args.model_checkpoint = preferred
            print(f"未指定 --model_checkpoint，自动使用: {args.model_checkpoint}")
        else:
            # 2) 其次优先选择 new_gnn_epoch_*.pt（更可能包含 feature_builder 权重）
            pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            new_style = [f for f in pt_files if f.startswith('new_gnn_epoch_')]

            candidates = new_style if new_style else pt_files
            if candidates:
                candidates.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
                args.model_checkpoint = os.path.join(ckpt_dir, candidates[0])
                print(f"未指定 --model_checkpoint，自动使用: {args.model_checkpoint}")
            else:
                print("未找到可用的模型权重文件，请通过 --model_checkpoint 指定！")
                args.model_checkpoint = None

    if not args.knowledge_base:
        if os.path.exists(Config.KNOWLEDGE_BASE_DIR):
            args.knowledge_base = Config.KNOWLEDGE_BASE_DIR
            print(f"未指定 --knowledge_base，自动使用: {args.knowledge_base}")
    
    # 创建输出目录
    Config.create_dirs()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 运行推理
    if args.mode == 'single':
        if not args.image_path:
            print("❌ 错误: single模式需要指定 --image_path")
            return
        
        if not os.path.exists(args.image_path):
            print(f"❌ 错误: 图像不存在: {args.image_path}")
            return
        
        run_single_inference(args)
    
    elif args.mode == 'batch':
        run_batch_inference(args)


if __name__ == "__main__":
    main()
