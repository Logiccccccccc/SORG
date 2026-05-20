import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM
try:
    from transformers import LlavaForConditionalGeneration, Blip2ForConditionalGeneration, Blip2Processor, InstructBlipForConditionalGeneration, InstructBlipProcessor
except ImportError:
    try:
        from transformers import LlavaNextForConditionalGeneration as LlavaForConditionalGeneration
    except ImportError:
        pass
    try:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
    except ImportError:
        pass
import json
import os
import re
import numpy as np
from PIL import Image
from config import Config
from model_core import PanopticSegmentationModule, RelationExtractor, CoreObjectSelector, CausalGNN, UncertaintyEstimator
from knowledge_builder import KnowledgeBaseBuilder

class MultimodalCaptionGenerator:

    def __init__(self, model_checkpoint=None, knowledge_base_path=None, score_weights=None):
        self.device = Config.DEVICE
        print('Loading visual model...')
        self.segmentation = PanopticSegmentationModule()
        print('Loading GNN model...')
        self.gnn = CausalGNN().to(self.device)
        if model_checkpoint and os.path.exists(model_checkpoint):
            print(f'Loading from {model_checkpoint} GNN weights...')
            checkpoint = torch.load(model_checkpoint, map_location=self.device)
            if 'gnn_state_dict' in checkpoint:
                self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.gnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.gnn.load_state_dict(checkpoint)
            if 'feature_builder_state_dict' in checkpoint:
                print('Loading Feature Builder (Fusion MLP) weights...')
                self.segmentation.feature_builder.load_state_dict(checkpoint['feature_builder_state_dict'])
            else:
                print('Warning: Feature Builder weights were not found in the checkpoint; random initialization will be used.')
        self.segmentation.eval()
        self.gnn.eval()
        print('Loading knowledge base...')
        self.kb_builder = KnowledgeBaseBuilder()
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            self.kb_builder.load_knowledge_base(knowledge_base_path)
        print('Loading relation extraction and core object selection modules...')
        self.relation_extractor = RelationExtractor().to(self.device)
        if score_weights is None:
            score_weights = (0.4, 0.3, 0.3)
        self.core_selector = CoreObjectSelector(weights=score_weights)
        self.uncertainty_estimator = UncertaintyEstimator()
        self.model_type = getattr(Config, 'MODEL_TYPE', 'llava')
        print(f'Loading VLM model: {self.model_type}...')
        if self.model_type == 'minigpt4':
            print(f'Loading MiniGPT-4/BLIP-2 from {Config.MINIGPT4_MODEL_PATH}...')
            self.vlm_processor = Blip2Processor.from_pretrained(Config.MINIGPT4_MODEL_PATH)
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(Config.MINIGPT4_MODEL_PATH, torch_dtype=dtype).to(self.device)
        elif self.model_type == 'instructblip':
            print(f'Loading InstructBLIP from {Config.INSTRUCTBLIP_MODEL_PATH}...')
            self.vlm_processor = InstructBlipProcessor.from_pretrained(Config.INSTRUCTBLIP_MODEL_PATH)
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(Config.INSTRUCTBLIP_MODEL_PATH, torch_dtype=dtype).to(self.device)
        elif self.model_type == 'qwen':
            print(f'Loading Qwen-VL from {Config.QWEN_MODEL_PATH}...')
            self.vlm_processor = AutoProcessor.from_pretrained(Config.QWEN_MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28, trust_remote_code=True)
            if hasattr(self.vlm_processor, 'tokenizer'):
                self.vlm_processor.tokenizer.padding_side = 'left'
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            try:
                from transformers import AutoModelForVision2Seq
                self.vlm_model = AutoModelForVision2Seq.from_pretrained(Config.QWEN_MODEL_PATH, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
            except Exception as e:
                print(f'Warning: AutoModelForVision2Seq failed ({e}), trying AutoModelForCausalLM...')
                try:
                    self.vlm_model = AutoModelForCausalLM.from_pretrained(Config.QWEN_MODEL_PATH, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
                except Exception as e2:
                    print(f'Warning: AutoModelForCausalLM failed ({e2})...')
                    self.vlm_model = AutoModelForCausalLM.from_pretrained(Config.QWEN_MODEL_PATH, torch_dtype=dtype, trust_remote_code=True).to(self.device)
        else:
            print('Loading LLaVA model. This may take a few minutes...')
            self.llava_processor = AutoProcessor.from_pretrained(Config.LLAVA_MODEL_PATH)
            self.vlm_processor = self.llava_processor
            print(f'Loading model to device: {self.device}')
            if self.device == 'cuda':
                self.llava_model = LlavaForConditionalGeneration.from_pretrained(Config.LLAVA_MODEL_PATH, torch_dtype=torch.float16).to(self.device)
            else:
                self.llava_model = LlavaForConditionalGeneration.from_pretrained(Config.LLAVA_MODEL_PATH).to(self.device)
            self.vlm_model = self.llava_model
        self.vlm_model.eval()
        print('Model initialization completed.')

    def extract_visual_features(self, image):
        self.segmentation.to(self.device)
        try:
            objects_batch, features_batch = self.segmentation([image])
        finally:
            self.segmentation.cpu()
            torch.cuda.empty_cache()
        objects = objects_batch[0]
        node_features = features_batch[0]
        relations_batch = self.relation_extractor.extract_spatial_relations([objects])
        relations = relations_batch[0]
        core_indices = self.core_selector.select_core_objects(objects, relations)
        return {'objects': objects, 'relations': relations, 'core_indices': core_indices, 'node_features': node_features}

    def perform_causal_reasoning(self, visual_features, use_uncertainty=True):
        objects = visual_features['objects']
        relations = visual_features['relations']
        core_indices = visual_features['core_indices']
        node_features = visual_features['node_features']
        if node_features.device != self.device:
            node_features = node_features.to(self.device)
        edges = []
        edge_feats = []
        for subj, rel, obj, conf in relations:
            edges.append([subj, obj])
            edge_feats.append([conf])
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
            edge_features = torch.tensor(edge_feats, dtype=torch.float).to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            edge_features = torch.empty((0, 1), dtype=torch.float).to(self.device)
        with torch.no_grad():
            embeddings, cooccur, causal = self.gnn(node_features, edge_index, edge_features)
        if use_uncertainty:
            uncertainty_result = self.uncertainty_estimator.mc_dropout_inference(self.gnn, node_features, edge_index, edge_features)
        else:
            uncertainty_result = {'cooccurrence': cooccur, 'causality': causal, 'uncertainty_cooccur': torch.zeros_like(cooccur), 'uncertainty_causal': torch.zeros_like(causal)}
        existence_probs = self.uncertainty_estimator.compute_existence_probability(core_indices, objects, uncertainty_result['cooccurrence'], uncertainty_result['causality'])
        return {'embeddings': embeddings, 'cooccurrence': uncertainty_result['cooccurrence'], 'causality': uncertainty_result['causality'], 'existence_probs': existence_probs, 'uncertainty_cooccur': uncertainty_result['uncertainty_cooccur'], 'uncertainty_causal': uncertainty_result['uncertainty_causal']}

    def construct_visual_context(self, visual_features, reasoning_result):
        objects = visual_features['objects']
        core_indices = visual_features['core_indices']
        existence_probs = reasoning_result['existence_probs']

        def _is_background_label(label):
            if getattr(Config, 'VISUAL_CONTEXT_FILTER_STUFF_MERGED', True):
                if 'merged' in label or 'stuff' in label or 'other' in label:
                    return True
            if getattr(Config, 'VISUAL_CONTEXT_FILTER_KEYWORDS', True):
                background_keywords = getattr(Config, 'VISUAL_CONTEXT_BACKGROUND_KEYWORDS', ('wall', 'floor', 'ceiling', 'pavement', 'road', 'grass', 'sky', 'building', 'mountain', 'dirt', 'curtain', 'rug', 'banner'))
                return any((k in label for k in background_keywords))
            return False
        core_objects = []
        seen_labels = set()
        ordered_core = list(core_indices)
        for idx in ordered_core:
            if idx < 0 or idx >= len(objects):
                continue
            obj = objects[idx]
            label = obj.get('label', '')
            if not label:
                continue
            if _is_background_label(label):
                continue
            label_l = label.lower()
            if label_l in seen_labels:
                continue
            seen_labels.add(label_l)
            core_objects.append(f"{label} (confidence: {obj.get('score', 0):.2f})")
        inferred_objects = []
        threshold = Config.INFERENCE_CONFIDENCE_THRESHOLD
        candidates = []
        for idx, prob in enumerate(existence_probs):
            if idx in core_indices:
                continue
            p = float(prob)
            if p > threshold:
                candidates.append((idx, p))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = getattr(Config, 'MAX_INFERRED_OBJECTS', 5)
        inferred_seen = set(seen_labels)
        for idx, prob in candidates:
            if idx < 0 or idx >= len(objects):
                continue
            label = objects[idx].get('label', '')
            if not label:
                continue
            label_l = label.lower()
            if label_l in inferred_seen:
                continue
            if _is_background_label(label):
                continue
            inferred_seen.add(label_l)
            inferred_objects.append(f'{label} (inferred probability: {prob:.2f})')
            if len(inferred_objects) >= top_k:
                break
        core_text = ', '.join(core_objects) if core_objects else ''
        inferred_text = ', '.join(inferred_objects) if inferred_objects else ''
        return f'Core objects:\n{core_text}\n\nPossible objects:\n{inferred_text}'

    def generate_caption_with_vlm(self, image, visual_context, baseline_mode=False):
        if self.model_type == 'qwen':
            if baseline_mode or not visual_context:
                prompt_content = 'Describe this image concisely.'
            else:
                prompt_content = f'Write one factual caption grounded in the image.\nThe lists below are constraints on which object NAMES you may mention (they are not a full description).\n\nRules:\n1) Focus on the image content.\n2) Object nouns: mention ONLY object names that appear under "Core objects" or "Possible objects".\n3) You MAY add attributes/actions/relations (e.g., colors, holding, on/next to) ONLY if they do not introduce any new object nouns.\n\n{visual_context}'
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt_content}]}]
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.vlm_processor(text=[text], images=[image], padding=True, return_tensors='pt')
            inputs = inputs.to(self.vlm_model.device)
            with torch.inference_mode():
                generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = self.vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return caption
        if self.model_type in ['minigpt4', 'instructblip']:
            if baseline_mode or not visual_context:
                if self.model_type == 'instructblip':
                    prompt = 'Describe this image concisely.'
                else:
                    prompt = 'Question: Describe this image concisely. Answer:'
            elif self.model_type == 'instructblip':
                prompt = f'[Context]\nThe following is a list of object concepts relevant to the image:\n{visual_context}\n\n[Instruction]\nWrite one short, factual caption for the image.\n- Use the concepts from the list above if they appear in the image.\n- Do NOT simply copy the list.\n- Write a natural sentence describing the scene.\nCaption:'
            else:
                prompt = f'Question: Write one factual caption grounded in the image.\nThe lists below are constraints on which object NAMES you may mention (they are not a full description).\n\nRules:\n1) Focus on the image content.\n2) Object nouns: mention ONLY object names that appear under "Core objects" or "Possible objects".\n3) You MAY add attributes/actions/relations (e.g., colors, holding, on/next to) ONLY if they do not introduce any new object nouns.\n\n{visual_context}\nAnswer:'
            inputs = self.vlm_processor(images=image, text=prompt, return_tensors='pt').to(self.device)
            with torch.inference_mode():
                output = self.vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=5, min_length=1, top_p=0.9, repetition_penalty=1.5, length_penalty=1.0, temperature=1)
            caption = self.vlm_processor.decode(output[0], skip_special_tokens=True)
            if caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            if 'Answer:' in caption:
                caption = caption.split('Answer:')[-1].strip()
            caption = caption.strip()
            return caption
        if baseline_mode or not visual_context:
            prompt = 'USER: <image>\nDescribe this image concisely.\nASSISTANT:'
        else:
            prompt = f'USER: <image>\nWrite one factual caption grounded in the image.\nThe lists below are constraints on which object NAMES you may mention (they are not a full description).\n\nRules:\n1) Focus on the image content.\n2) Object nouns: mention ONLY object names that appear under "Core objects" or "Possible objects".\n3) You MAY add attributes/actions/relations (e.g., colors, holding, on/next to) ONLY if they do not introduce any new object nouns.\n\n{visual_context}\nASSISTANT:'
        inputs = self.vlm_processor(text=prompt, images=image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.vlm_model.generate(**inputs, max_new_tokens=96, do_sample=False, num_beams=3, early_stopping=True, no_repeat_ngram_size=3, repetition_penalty=1.1)
        caption = self.vlm_processor.decode(output[0], skip_special_tokens=True)
        if 'ASSISTANT:' in caption:
            caption = caption.split('ASSISTANT:')[-1].strip()
        return caption

    def generate(self, image_path, ablation_mode='complete', baseline_mode=False):
        if baseline_mode:
            ablation_mode = 'baseline'
        image = Image.open(image_path).convert('RGB')
        filename = os.path.basename(image_path)
        match = re.search('(\\d+)', filename)
        image_id = int(match.group(1)) if match else 0
        if ablation_mode == 'baseline':
            print('Steps 1-3: skipped in baseline mode')
            visual_features = {'objects': [], 'core_indices': []}
            reasoning_result = {'existence_probs': []}
            visual_context = ''
        else:
            print('Step 1: extracting visual features...')
            visual_features = self.extract_visual_features(image)
            if ablation_mode == 'no_clip':
                n_objects = len(visual_features['objects'])
                visual_features['node_features'] = torch.randn(n_objects, Config.GNN_HIDDEN_DIM).to(self.device)
            print('Step 2: running causal reasoning...')
            if ablation_mode == 'no_reasoning':
                reasoning_result = {'existence_probs': [0.0] * len(visual_features['objects']), 'cooccurrence': None, 'causality': None}
            else:
                if ablation_mode == 'no_relations':
                    visual_features['relations'] = []
                use_uncertainty = ablation_mode != 'no_uncertainty'
                reasoning_result = self.perform_causal_reasoning(visual_features, use_uncertainty=use_uncertainty)
            print('Step 3: building visual context...')
            visual_context = self.construct_visual_context(visual_features, reasoning_result)
        print('Step 4: generating caption...')
        caption = self.generate_caption_with_vlm(image, visual_context)
        print(f'Processing finished: {image_path}')
        objects = visual_features.get('objects', [])
        core_indices = set(visual_features.get('core_indices', []))
        raw_probs = reasoning_result.get('existence_probs', [])
        prob_values = [float(p) for p in raw_probs]
        detected_objects = [{'label': obj.get('label', ''), 'score': float(obj.get('score', 0.0)), 'is_core': idx in core_indices} for idx, obj in enumerate(objects)]
        inferred_objects = []
        threshold = Config.INFERENCE_CONFIDENCE_THRESHOLD
        for idx, obj in enumerate(objects):
            if idx in core_indices or idx >= len(prob_values):
                continue
            if prob_values[idx] > threshold:
                inferred_objects.append({'label': obj.get('label', ''), 'existence_prob': prob_values[idx]})
        return {'image_id': image_id, 'caption': caption, 'visual_context': visual_context, 'detected_objects': detected_objects, 'inferred_objects': inferred_objects, 'visual_features': {'objects': detected_objects}, 'reasoning_result': {'existence_probs': prob_values}}
