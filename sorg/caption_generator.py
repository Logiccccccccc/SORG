"""多模态描述生成模块"""
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM
try:
    from transformers import (
        LlavaForConditionalGeneration, 
        Blip2ForConditionalGeneration, Blip2Processor,
        InstructBlipForConditionalGeneration, InstructBlipProcessor
    )
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
from model_core import (
    PanopticSegmentationModule,
    RelationExtractor,
    CoreObjectSelector,
    CausalGNN,
    UncertaintyEstimator
)
from knowledge_builder import KnowledgeBaseBuilder


class MultimodalCaptionGenerator:
    """多模态描述生成器"""
    
    def __init__(self, model_checkpoint=None, knowledge_base_path=None, score_weights=None):
        """
        初始化描述生成器
        Args:
            model_checkpoint: GNN模型检查点路径
            knowledge_base_path: 知识库路径
            score_weights: 核心对象评分权重 (w_conf, w_area, w_cent)
        """
        self.device = Config.DEVICE
        
        print("加载视觉模型...")
        # Reduce VRAM usage: Do not move to GPU immediately. Move on demand.
        self.segmentation = PanopticSegmentationModule() # .to(self.device)
        
        print("加载GNN模型...")
        self.gnn = CausalGNN().to(self.device)
        

        # 加载训练好的GNN权重
        if model_checkpoint and os.path.exists(model_checkpoint):
            print(f"从 {model_checkpoint} 加载GNN权重...")
            checkpoint = torch.load(model_checkpoint, map_location=self.device)
            if 'gnn_state_dict' in checkpoint:
                self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.gnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.gnn.load_state_dict(checkpoint)

            # === 新增：加载 Feature Builder (MLP) 权重 ===
            if 'feature_builder_state_dict' in checkpoint:
                print("加载 Feature Builder (Fusion MLP) 权重...")
                self.segmentation.feature_builder.load_state_dict(checkpoint['feature_builder_state_dict'])

            else:
                print("⚠ 警告: 权重文件中未找到 Feature Builder 权重，将使用随机初始化！")

        # 推理阶段：确保视觉特征构建器与GNN处于 eval 模式（Dropout 关闭）
        # MC Dropout 只在 UncertaintyEstimator.mc_dropout_inference 中临时开启
        self.segmentation.eval()
        self.gnn.eval()
        print("加载知识库...")
        self.kb_builder = KnowledgeBaseBuilder()
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            self.kb_builder.load_knowledge_base(knowledge_base_path)
        
        # 关系提取和核心对象选择 - 关键修复
        print("加载关系提取和核心对象选择模块...")
        self.relation_extractor = RelationExtractor().to(self.device)
        
        if score_weights is None:
            score_weights = (0.4, 0.3, 0.3)
        self.core_selector = CoreObjectSelector(weights=score_weights)
        
        # 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator()
        
        self.model_type = getattr(Config, 'MODEL_TYPE', 'llava')
        print(f"Loading VLM model: {self.model_type}...")

        if self.model_type == 'minigpt4':
            print(f"Loading MiniGPT-4/BLIP-2 from {Config.MINIGPT4_MODEL_PATH}...")
            self.vlm_processor = Blip2Processor.from_pretrained(Config.MINIGPT4_MODEL_PATH)
            
            # Use float16 for GPU
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            self.vlm_model = Blip2ForConditionalGeneration.from_pretrained(
                Config.MINIGPT4_MODEL_PATH,
                torch_dtype=dtype
            ).to(self.device)
            
        elif self.model_type == 'instructblip':
            print(f"Loading InstructBLIP from {Config.INSTRUCTBLIP_MODEL_PATH}...")
            self.vlm_processor = InstructBlipProcessor.from_pretrained(Config.INSTRUCTBLIP_MODEL_PATH)
            
            # Use float16 for GPU
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(
                Config.INSTRUCTBLIP_MODEL_PATH,
                torch_dtype=dtype
            ).to(self.device)

        elif self.model_type == 'qwen':
            print(f"Loading Qwen-VL from {Config.QWEN_MODEL_PATH}...")
            # 尝试加载 Qwen 模型
            self.vlm_processor = AutoProcessor.from_pretrained(
                Config.QWEN_MODEL_PATH, 
                min_pixels=256*28*28, 
                max_pixels=1280*28*28,
                trust_remote_code=True
            )
            # Fix for batch generation warning: decoder-only architecture requires left padding
            if hasattr(self.vlm_processor, 'tokenizer'):
                self.vlm_processor.tokenizer.padding_side = 'left'
            
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            
            # 使用 device_map="auto" 自动分配显存，或者手动 .to(device)
            # 这里如果不安装 accelerate 可能需要手动 .to(device)
            # 为了稳健，我们尝试使用 device_map
            try:
                # Qwen2-VL/Qwen2.5-VL/Qwen3-VL 通常需要使用 AutoModelForVision2Seq
                # 某些 Qwen 多模态模型使用 AutoModelForCausalLM，但最新的 VL 模型多使用 Vision2Seq
                from transformers import AutoModelForVision2Seq
                self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                    Config.QWEN_MODEL_PATH,
                    torch_dtype=dtype,
                    device_map="auto", 
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: AutoModelForVision2Seq failed ({e}), trying AutoModelForCausalLM...")
                try:
                    self.vlm_model = AutoModelForCausalLM.from_pretrained(
                        Config.QWEN_MODEL_PATH,
                        torch_dtype=dtype,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except Exception as e2:
                     print(f"Warning: AutoModelForCausalLM failed ({e2})...")
                     # Fallback to manual device placement
                     self.vlm_model = AutoModelForCausalLM.from_pretrained(
                        Config.QWEN_MODEL_PATH,
                        torch_dtype=dtype,
                        trust_remote_code=True
                    ).to(self.device)
            
        else:
            # Default to LLaVA
            print("加载LLaVA模型(这可能需要几分钟)...")
            self.llava_processor = AutoProcessor.from_pretrained(Config.LLAVA_MODEL_PATH)
            self.vlm_processor = self.llava_processor # Alias
            
            # 根据设备加载模型
            print(f"将模型加载到设备: {self.device}")
            if self.device == 'cuda':
                self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                    Config.LLAVA_MODEL_PATH,
                    torch_dtype=torch.float16,
                ).to(self.device)
            else:
                self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                    Config.LLAVA_MODEL_PATH,
                ).to(self.device)
            self.vlm_model = self.llava_model # Alias
            
        self.vlm_model.eval()
        
        print("✓ 模型初始化完成")
    
    # def extract_visual_features(self, image):
    #     """提取图像的视觉特征"""
    #     objects_list = self.segmentation([image])
    #     objects = objects_list[0]
        
    #     relations = self.relation_extractor.extract_spatial_relations(objects)
    #     core_indices = self.core_selector.select_core_objects(objects, relations)
        
    #     return {
    #         'objects': objects,
    #         'relations': relations,
    #         'core_indices': core_indices
    #     }
    

    def extract_visual_features(self, image):
        """提取图像的视觉特征 (已修复以支持 CLIP 和 Batch 结构)"""
        # 1. 获取分割结果和 CLIP 特征
        # --- OOM Optimization: Move segmentation to GPU temporarily ---
        self.segmentation.to(self.device)
        try:
            # 注意：segmentation 现在返回 (objects_list, features_list)
            objects_batch, features_batch = self.segmentation([image])
        finally:
            # --- OOM Optimization: Move back to CPU immediately ---
            self.segmentation.cpu()
            torch.cuda.empty_cache()
        
        # 因为我们是单张处理，取第一个元素
        objects = objects_batch[0]       # List[Dict]
        node_features = features_batch[0] # Tensor [N, 512] (CLIP特征)
        
        # 2. 提取关系
        # relation_extractor 期望输入是 Batch 格式 [[obj1, obj2...]]
        # 返回也是 Batch 格式 [[(s,r,o,c)...]]
        relations_batch = self.relation_extractor.extract_spatial_relations([objects])
        relations = relations_batch[0]   # 取出当前图片的关系列表
        
        # 3. 选择核心对象
        core_indices = self.core_selector.select_core_objects(objects, relations)
        
        return {
            'objects': objects,
            'relations': relations,
            'core_indices': core_indices,
            'node_features': node_features  # <--- 新增：保存 CLIP 特征
        }

    # def perform_causal_reasoning(self, visual_features):
    #     """执行因果推理"""
    #     objects = visual_features['objects']
    #     relations = visual_features['relations']
    #     core_indices = visual_features['core_indices']
        
    #     n_objects = len(objects)
    #     node_features = torch.randn(n_objects, Config.GNN_HIDDEN_DIM).to(self.device)
        
    #     edges = []
    #     edge_feats = []
    #     for subj, rel, obj, conf in relations:
    #         edges.append([subj, obj])
    #         edge_feats.append([conf])
        
    #     if edges:
    #         edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
    #         edge_features = torch.tensor(edge_feats, dtype=torch.float).to(self.device)
    #     else:
    #         edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
    #         edge_features = torch.empty((0, 1), dtype=torch.float).to(self.device)
        
    #     with torch.no_grad():
    #         embeddings, cooccur, causal = self.gnn(node_features, edge_index, edge_features)
        
    #     uncertainty_result = self.uncertainty_estimator.mc_dropout_inference(
    #         self.gnn, node_features, edge_index, edge_features
    #     )
        
    #     existence_probs = self.uncertainty_estimator.compute_existence_probability(
    #         core_indices, objects, uncertainty_result['cooccurrence'], 
    #         uncertainty_result['causality']
    #     )
        
    #     return {
    #         'embeddings': embeddings,
    #         'cooccurrence': uncertainty_result['cooccurrence'],
    #         'causality': uncertainty_result['causality'],
    #         'existence_probs': existence_probs,
    #         'uncertainty_cooccur': uncertainty_result['uncertainty_cooccur'],
    #         'uncertainty_causal': uncertainty_result['uncertainty_causal']
    #     }

    def perform_causal_reasoning(self, visual_features, use_uncertainty=True):
        """执行因果推理 (已修复以使用真实特征)"""
        objects = visual_features['objects']
        relations = visual_features['relations']
        core_indices = visual_features['core_indices']
        
        # --- 关键修改：使用 CLIP 特征，而不是随机噪声 ---
        # 旧代码: node_features = torch.randn(...) 
        # 新代码:
        node_features = visual_features['node_features'] 
        
        # 确保特征在正确的设备上
        if node_features.device != self.device:
            node_features = node_features.to(self.device)
            
        # 构建边索引 (Edge Index)
        edges = []
        edge_feats = []
        
        # 注意：这里的 relations 已经是列表 [(subj, rel, obj, conf), ...]
        # 不会再报 "too many values to unpack" 错误了
        for subj, rel, obj, conf in relations:
            edges.append([subj, obj])
            edge_feats.append([conf])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
            edge_features = torch.tensor(edge_feats, dtype=torch.float).to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            edge_features = torch.empty((0, 1), dtype=torch.float).to(self.device)
        
        # GNN 推理
        with torch.no_grad():
            embeddings, cooccur, causal = self.gnn(node_features, edge_index, edge_features)
        
        if use_uncertainty:
            # 不确定性估计
            uncertainty_result = self.uncertainty_estimator.mc_dropout_inference(
                self.gnn, node_features, edge_index, edge_features
            )
        else:
            uncertainty_result = {
                'cooccurrence': cooccur,
                'causality': causal,
                'uncertainty_cooccur': torch.zeros_like(cooccur),
                'uncertainty_causal': torch.zeros_like(causal)
            }
        
        existence_probs = self.uncertainty_estimator.compute_existence_probability(
            core_indices, objects, uncertainty_result['cooccurrence'], 
            uncertainty_result['causality']
        )
        
        return {
            'embeddings': embeddings,
            'cooccurrence': uncertainty_result['cooccurrence'],
            'causality': uncertainty_result['causality'],
            'existence_probs': existence_probs,
            'uncertainty_cooccur': uncertainty_result['uncertainty_cooccur'],
            'uncertainty_causal': uncertainty_result['uncertainty_causal']
        }
    def construct_visual_context(self, visual_features, reasoning_result):
        """构建视觉上下文提示"""
        objects = visual_features['objects']
        core_indices = visual_features['core_indices']
        existence_probs = reasoning_result['existence_probs']

        def _is_background_label(label):           # maskformer 的 stuff/merged 类非常多，容易把 prompt 带偏；但过滤太狠会降低描述质量。
            # 通过 Config 开关支持对照实验。
            if getattr(Config, 'VISUAL_CONTEXT_FILTER_STUFF_MERGED', True):
                if 'merged' in label or 'stuff' in label or 'other' in label:
                    return True
            if getattr(Config, 'VISUAL_CONTEXT_FILTER_KEYWORDS', True):
                background_keywords = getattr(
                    Config,
                    'VISUAL_CONTEXT_BACKGROUND_KEYWORDS',
                    (
                        'wall', 'floor', 'ceiling', 'pavement', 'road', 'grass', 'sky', 'building',
                        'mountain', 'dirt', 'curtain', 'rug', 'banner'
                    )
                )
                return any(k in label for k in background_keywords)
            return False
        
        # 1) 核心对象：去重 + 仅保留非背景对象（不再回退到背景）
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
            core_objects.append(f"{label} (置信度: {obj.get('score', 0):.2f})")
        
        # 2) 推断对象：更严格阈值 + 去重 + 过滤背景
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
            inferred_objects.append(f"{label} (推断概率: {prob:.2f})")
            if len(inferred_objects) >= top_k:
                break

        # 按用户要求：不再包含 relations，回退为“之前的样子”的文本结构
        core_text = ", ".join(core_objects) if core_objects else ""
        inferred_text = ", ".join(inferred_objects) if inferred_objects else ""
        return (
            "检测到的核心对象:\n"
            f"{core_text}"
            "\n\n"
            "可能存在的对象:\n"
            f"{inferred_text}"
        )
    
    def generate_caption_with_vlm(self, image, visual_context, baseline_mode=False):
        """使用 VLM (LLaVA, MiniGPT-4, InstructBLIP, Qwen) 生成描述"""

        # === Qwen Path ===
        if self.model_type == 'qwen':
            # Prompt Construction
            if baseline_mode or not visual_context:
                prompt_content = "Describe this image concisely."
            else:
                prompt_content = f"""Write one factual caption grounded in the image.
The lists below are constraints on which object NAMES you may mention (they are not a full description).

Rules:
1) Focus on the image content.
2) Object nouns: mention ONLY object names that appear under "检测到的核心对象" or "可能存在的对象".
3) You MAY add attributes/actions/relations (e.g., colors, holding, on/next to) ONLY if they do not introduce any new object nouns.

{visual_context}"""
            
            # Use AutoProcessor / Qwen2VLProcessor logic
            # Messages format for apply_chat_template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_content},
                    ],
                }
            ]
            
            # Generate input text with special tokens
            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs (Image + Text)
            inputs = self.vlm_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vlm_model.device)

            with torch.inference_mode():
                generated_ids = self.vlm_model.generate(
                    **inputs, 
                    max_new_tokens=128
                )
            
            # Decode output (skipping input tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = self.vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return caption

        # === MiniGPT-4 / BLIP-2 / InstructBLIP Path ===
        if self.model_type in ['minigpt4', 'instructblip']:
            if baseline_mode or not visual_context:
                if self.model_type == 'instructblip':
                    # InstructBLIP 不需要 Question: Answer: 格式，直接给指令更好
                    prompt = "Describe this image concisely."
                else:
                    prompt = "Question: Describe this image concisely. Answer:"
            else:
                if self.model_type == 'instructblip':
                    # InstructBLIP 对于列表非常敏感，容易直接复述列表。
                    # 解决方案：
                    # 1. 结构化 Prompt，把指令放在最后 (Recency Bias)
                    # 2. 也是最重要的：明确禁止复述列表 (Negative Constraints)
                    prompt = f"""[Context]
The following is a list of object concepts relevant to the image:
{visual_context}

[Instruction]
Write one short, factual caption for the image.
- Use the concepts from the list above if they appear in the image.
- Do NOT simply copy the list.
- Write a natural sentence describing the scene.
Caption:"""
                else:
                    # BLIP-2 (MiniGPT-4) 还是保持原有的 QA 格式
                    prompt = f"""Question: Write one factual caption grounded in the image.
The lists below are constraints on which object NAMES you may mention (they are not a full description).

Rules:
1) Focus on the image content.
2) Object nouns: mention ONLY object names that appear under "检测到的核心对象" or "可能存在的对象".
3) You MAY add attributes/actions/relations (e.g., colors, holding, on/next to) ONLY if they do not introduce any new object nouns.

{visual_context}
Answer:"""

            # BLIP-2 / MiniGPT-4 / InstructBLIP logic
            inputs = self.vlm_processor(images=image, text=prompt, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                # InstructBLIP 推荐使用 longer max_new_tokens for detailed captions, 
                # but here we want concise one.
                output = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=5, # InstructBLIP works better with 5 beams
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )

            caption = self.vlm_processor.decode(output[0], skip_special_tokens=True)
            
            # --- 修复: 移除 caption 中可能包含的 Prompt ---
            # 有些模型(如 InstructBLIP-Vicuna)是 Causal LM，输出会包含 Input Prompt。
            # 简单的修复：如果 caption 以 prompt 开头，将其移除。
            
            # 1. 尝试基于字符串移除 (简单直接)
            if caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
            
            # 2. 尝试基于 Answer: 标记移除 (针对 BLIP-2/MiniGPT-4)
            if "Answer:" in caption:
                caption = caption.split("Answer:")[-1].strip()
                
            # 3. 再次清洗可能残留的 Newlines
            caption = caption.strip()

            return caption

        # === LLaVA Path ===
        if baseline_mode or not visual_context:
            prompt = "USER: <image>\nDescribe this image concisely.\nASSISTANT:"
        else:
            prompt = f"""USER: <image>
Write one factual caption grounded in the image.
The lists below are constraints on which object NAMES you may mention (they are not a full description).

Rules:
1) Focus on the image content.
2) Object nouns: mention ONLY object names that appear under "检测到的核心对象" or "可能存在的对象".
3) You MAY add attributes/actions/relations (e.g., colors, holding, on/next to) ONLY if they do not introduce any new object nouns.

{visual_context}
ASSISTANT:"""
        
        inputs = self.vlm_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            output = self.vlm_model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.10
            )
        
        caption = self.vlm_processor.decode(output[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()
        
        return caption
    
    def generate(self, image_path, ablation_mode='complete', baseline_mode=False):
        """完整的生成流程"""
        # 兼容旧参数
        if baseline_mode:
            ablation_mode = 'baseline'

        image = Image.open(image_path).convert('RGB')
        
        # 提取 image_id
        filename = os.path.basename(image_path)
        match = re.search(r'(\d+)', filename)
        image_id = int(match.group(1)) if match else 0
        
        if ablation_mode == 'baseline':
            print("步骤1-3: 跳过 (Baseline模式)")
            visual_features = {'objects': [], 'core_indices': []}
            reasoning_result = {'existence_probs': []}
            visual_context = ""
        else:
            print("步骤1: 提取视觉特征...")
            visual_features = self.extract_visual_features(image)
            
            # Ablation: No CLIP (使用随机特征替代 CLIP 特征)
            if ablation_mode == 'no_clip':
                n_objects = len(visual_features['objects'])
                visual_features['node_features'] = torch.randn(n_objects, Config.GNN_HIDDEN_DIM).to(self.device)

            print("步骤2: 执行因果推理...")
            if ablation_mode == 'no_reasoning':
                # 跳过推理，不产生推断对象
                reasoning_result = {
                    'existence_probs': [0.0] * len(visual_features['objects']),
                    'cooccurrence': None,
                    'causality': None
                }
            else:
                # Ablation: No Relations (清空关系)
                if ablation_mode == 'no_relations':
                    visual_features['relations'] = []
                
                # Ablation: No Uncertainty
                use_uncertainty = (ablation_mode != 'no_uncertainty')
                
                reasoning_result = self.perform_causal_reasoning(visual_features, use_uncertainty=use_uncertainty)
            
            print("步骤3: 构建视觉上下文...")
            visual_context = self.construct_visual_context(visual_features, reasoning_result)
        
        print("步骤4: 生成描述...")
        caption = self.generate_caption_with_vlm(image, visual_context)
        
        print(f"处理完成: {image_path}")
        
        return {
            'caption': caption,
            'visual_features': {
                'objects': [
                    {
                        'label': obj['label'],
                        'score': float(obj['score']),
                        'is_core': idx in visual_features['core_indices']
                    }
                    for idx, obj in enumerate(visual_features['objects'])
                ]
            },
            'reasoning_result': {
                'existence_probs': [float(p) for p in reasoning_result['existence_probs']]
            },
            'visual_context': visual_context
        }