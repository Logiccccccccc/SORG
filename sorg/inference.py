import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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
    filename = os.path.basename(image_path)
    match = re.search('(\\d+)', filename)
    return int(match.group(1)) if match else 0

def run_single_inference(args):
    print('=' * 60)
    print('Initializing generator...')
    if args.baseline and args.ablation_mode == 'complete':
        args.ablation_mode = 'baseline'
    print(f'Mode: {args.ablation_mode}')
    if args.ablation_mode == 'baseline':
        print('  - LLaVA only; skipping GNN and knowledge base.')
    elif args.ablation_mode == 'no_reasoning':
        print('  - Visual detection only; skipping GNN reasoning.')
    elif args.ablation_mode == 'no_relations':
        print('  - Spatial relations disabled; using an edgeless graph.')
    elif args.ablation_mode == 'no_uncertainty':
        print('  - Uncertainty estimation disabled; using one forward pass.')
    elif args.ablation_mode == 'no_clip':
        print('  - CLIP features disabled; using random features.')
    elif args.ablation_mode.startswith('selector_'):
        print(f'  - Core object selection ablation: {args.ablation_mode}')
    else:
        print('  - Complete mode: using GNN, spatial relations, CLIP features, and uncertainty enhancement.')
    print('=' * 60)
    generator = MultimodalCaptionGenerator(model_checkpoint=args.model_checkpoint, knowledge_base_path=args.knowledge_base, score_weights=args.score_weights_tuple)
    print('\n' + '=' * 60)
    print(f'Processing image: {args.image_path}')
    print('=' * 60)
    try:
        result = generator.generate(args.image_path, ablation_mode=args.ablation_mode)
        print('\n' + '=' * 60)
        print('Generation Result')
        print('=' * 60)
        print(f'\nGenerated caption')
        print(result['caption'])
        print(f'\nVisual context')
        print(result['visual_context'])
        print(f"\nDetected objects({len(result['detected_objects'])} )")
        for obj in result['detected_objects'][:10]:
            core_tag = ' [core]' if obj['is_core'] else ''
            print(f"  - {obj['label']}: {obj['score']:.2f}{core_tag}")
        if result['inferred_objects']:
            print(f"\nInferred objects({len(result['inferred_objects'])} )")
            for obj in result['inferred_objects'][:5]:
                print(f"  - {obj['label']}: {obj['existence_prob']:.2f}")
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f'\nResults saved to: {args.output_file}')
        return result
    except Exception as e:
        print(f'\nError: processing failed')
        print(f'   {str(e)}')
        import traceback
        traceback.print_exc()
        return None

def run_batch_inference(args):
    print('=' * 60)
    print('Batch inference mode')
    if args.baseline and args.ablation_mode == 'complete':
        args.ablation_mode = 'baseline'
    print(f'Mode: {args.ablation_mode}')
    if args.ablation_mode == 'baseline':
        print('  - LLaVA only; skipping GNN and knowledge base.')
    elif args.ablation_mode == 'no_reasoning':
        print('  - Visual detection only; skipping GNN reasoning.')
    elif args.ablation_mode == 'no_relations':
        print('  - Spatial relations disabled; using an edgeless graph.')
    elif args.ablation_mode == 'no_uncertainty':
        print('  - Uncertainty estimation disabled; using one forward pass.')
    elif args.ablation_mode == 'no_clip':
        print('  - CLIP features disabled; using random features.')
    elif args.ablation_mode.startswith('selector_'):
        print(f'  - Core object selection ablation: {args.ablation_mode}')
    else:
        print('  - Complete mode: using GNN, spatial relations, CLIP features, and uncertainty enhancement.')
    print('=' * 60)
    if args.annotation_file:
        print(f'Loading image list from annotation file: {args.annotation_file}')
        with open(args.annotation_file, 'r') as f:
            coco_data = json.load(f)
        image_paths = []
        missing_count = 0
        max_needed = args.max_samples if args.max_samples and args.max_samples > 0 else None
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
            print(f'Found {len(image_paths)} valid images with no max_samples limit')
        else:
            print(f'Target: {max_needed}; found: {len(image_paths)} valid images; skipped: {missing_count} missing images')
    elif args.image_list:
        print(f'Loading images from image list file: {args.image_list}')
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        image_paths = [p for p in image_paths if os.path.exists(p)]
        print(f'Found {len(image_paths)} valid images')
    elif args.image_dir:
        print(f'Loading images from directory: {args.image_dir}')
        image_paths = []
        for fname in os.listdir(args.image_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(args.image_dir, fname))
        if args.max_samples > 0:
            image_paths = image_paths[:args.max_samples]
        print(f'Found {len(image_paths)} images')
    else:
        print('Error: batch mode requires --annotation_file, --image_list, or --image_dir.')
        return None
    if not image_paths:
        print('Error: no images found.')
        return None
    print('\nInitializing generator...')
    generator = MultimodalCaptionGenerator(model_checkpoint=args.model_checkpoint, knowledge_base_path=args.knowledge_base, score_weights=args.score_weights_tuple)
    print(f'\nStarting to process {len(image_paths)} images...')
    results = []
    success_count = 0
    failed_count = 0
    for img_path in tqdm(image_paths, desc='Generating captions'):
        try:
            result = generator.generate(img_path, ablation_mode=args.ablation_mode)
            if 'image_id' not in result:
                result['image_id'] = extract_image_id(img_path)
            if 'detected_objects' not in result and 'visual_features' in result and ('objects' in result['visual_features']):
                result['detected_objects'] = result['visual_features']['objects']
            if 'inferred_objects' not in result:
                result['inferred_objects'] = []
            if 'caption' not in result or not result['caption']:
                result['caption'] = ''
            from collections import OrderedDict
            final_result = OrderedDict()
            final_result['image_id'] = result['image_id']
            final_result['image_path'] = img_path
            final_result['image_file'] = os.path.basename(img_path)
            final_result['caption'] = result['caption']
            final_result['visual_context'] = result.get('visual_context', '')
            final_result['detected_objects'] = result.get('detected_objects', [])
            final_result['inferred_objects'] = result.get('inferred_objects', [])
            if 'error' in result:
                final_result['error'] = result['error']
            results.append(final_result)
            success_count += 1
        except Exception as e:
            print(f'\nProcessing failed: {img_path}')
            print(f'  Error: {str(e)}')
            results.append({'image_id': extract_image_id(img_path), 'image_path': img_path, 'image_file': os.path.basename(img_path), 'caption': '', 'error': str(e)})
            failed_count += 1
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print('\n' + '=' * 60)
        print('Batch inference completed')
        print('=' * 60)
        print(f'Total: {len(image_paths)}')
        print(f'Succeeded: {success_count}')
        print(f'Failed: {failed_count}')
        print(f'\nResults saved to: {args.output_file}')
    return results

def main():
    parser = argparse.ArgumentParser(description='Image caption generation inference')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'], help='Inference mode: single or batch.')
    parser.add_argument('--image_path', type=str, help='Single image path for single mode.')
    parser.add_argument('--annotation_file', type=str, help='COCO annotation file for batch mode.')
    parser.add_argument('--image_dir', type=str, help='Image directory for batch mode.')
    parser.add_argument('--image_list', type=str, help='Image path list file for batch mode.')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of images to process')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='GNN model checkpoint path.')
    parser.add_argument('--knowledge_base', type=str, default=None, help='Knowledge base path.')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file path.')
    parser.add_argument('--baseline', action='store_true', help='Deprecated. Use --ablation_mode baseline instead.')
    parser.add_argument('--model_type', type=str, default=None, choices=['llava', 'minigpt4', 'instructblip', 'qwen'], help='Override MODEL_TYPE from the loaded YAML config.')
    parser.add_argument('--ablation_mode', type=str, default='complete', choices=['complete', 'baseline', 'no_reasoning', 'no_relations', 'no_uncertainty', 'no_clip', 'selector_confidence_only', 'selector_no_relations', 'selector_random'], help='Ablation mode: complete, baseline, no_reasoning, no_relations, no_uncertainty, no_clip, selector_confidence_only, selector_no_relations, selector_random')
    parser.add_argument('--score_weights', type=str, default=None, help='Core object score weights, format: "0.4,0.3,0.3"')
    parser.add_argument('--config', type=str, default=None, help='Path to the YAML inference config')
    args = parser.parse_args()
    selected_config = args.config
    if selected_config is None:
        selected_config = 'configs/infer_qwen.yaml' if args.model_type == 'qwen' else 'configs/infer_llava.yaml'
    Config.load_config(selected_config)
    if args.model_type:
        Config.MODEL_TYPE = args.model_type
        print(f'Model type forced by command line: {Config.MODEL_TYPE}')
    if args.score_weights:
        try:
            args.score_weights_tuple = tuple(map(float, args.score_weights.split(',')))
            if len(args.score_weights_tuple) != 3:
                raise ValueError('Must provide 3 weights')
        except Exception as e:
            print(f'Invalid score weights: {e}')
            return
    else:
        args.score_weights_tuple = None
    Config.create_dirs()
    if not args.model_checkpoint:
        ckpt_dir = Config.CHECKPOINT_DIR
        preferred = os.path.join(ckpt_dir, 'gnn_state_5000.pt')
        if os.path.exists(preferred):
            args.model_checkpoint = preferred
            print(f'No --model_checkpoint specified. Auto-selected: {args.model_checkpoint}')
        else:
            pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
            new_style = [f for f in pt_files if f.startswith('new_gnn_epoch_')]
            candidates = new_style if new_style else pt_files
            if candidates:
                candidates.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
                args.model_checkpoint = os.path.join(ckpt_dir, candidates[0])
                print(f'No --model_checkpoint specified. Auto-selected: {args.model_checkpoint}')
            else:
                print('No available model checkpoint was found. Please specify --model_checkpoint.')
                args.model_checkpoint = None
    if not args.knowledge_base:
        if os.path.exists(Config.KNOWLEDGE_BASE_DIR):
            args.knowledge_base = Config.KNOWLEDGE_BASE_DIR
            print(f'No --knowledge_base specified. Auto-selected: {args.knowledge_base}')
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.mode == 'single':
        if not args.image_path:
            print('Error: single mode requires --image_path')
            return
        if not os.path.exists(args.image_path):
            print(f'Error: image does not exist: {args.image_path}')
            return
        run_single_inference(args)
    elif args.mode == 'batch':
        run_batch_inference(args)
if __name__ == '__main__':
    main()
