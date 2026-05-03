import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import cdist
import math
from config import Config

class PanopticSegmentationModule(nn.Module):
    """全景分割模块：集成特征提取与融合"""

    class NodeFeatureBuilder(nn.Module):
        """内部类：节点特征构建器"""
        def __init__(self, device=Config.DEVICE):
            super().__init__()
            self.device = device
            
            # 1. 加载 CLIP (同时处理文本和图像，效率更高)
            print("Loading CLIP for Feature Extraction...")
            # 使用 base-patch32 速度较快，显存占用较小
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # 冻结 CLIP，只做特征提取
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # 维度定义 (CLIP Base 输出通常是 512)
            self.visual_dim = 512
            self.text_dim = 512
            self.pos_dim = 4
            self.out_dim = Config.GNN_HIDDEN_DIM
            
            # 2. 融合 MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.visual_dim + self.text_dim + self.pos_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, self.out_dim),
                nn.LayerNorm(self.out_dim) # 加个Norm更稳定
            )

        def forward(self, original_image, objects):
            """
            Args:
                original_image: PIL Image
                objects: list of dict (包含 label 和 calculated bbox)
            """
            if not objects:
                return torch.empty(0, self.out_dim).to(self.device)

            # --- 1. 批量准备文本特征 ---
            labels = [obj['label'] for obj in objects]
            text_inputs = self.clip_processor(text=labels, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_feats = self.clip_model.get_text_features(**text_inputs) # [N, 512]
            # 添加 epsilon 防止除零
            text_feats = text_feats / (text_feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            # --- 2. 批量准备视觉特征 (Crop + CLIP) ---
            crops = []
            valid_indices = []
            W, H = original_image.size
            # 防止 W, H 为 0
            W = max(W, 1)
            H = max(H, 1)
            
            # for idx, obj in enumerate(objects):
            #     x, y, w, h = obj['bbox']
            #     # 坐标边界保护
            #     x1, y1 = max(0, x), max(0, y)
            #     x2, y2 = min(W, x+w), min(H, y+h)
                
            #     if x2 > x1 and y2 > y1:
            #         crop = original_image.crop((x1, y1, x2, y2))
            #         # 保证为RGB格式
            #         if crop.mode != 'RGB':
            #             crop = crop.convert('RGB')
            #         crops.append(crop)
            #         valid_indices.append(idx)


            # ... 前面的代码 ...
            
            for idx, obj in enumerate(objects):
                x, y, w, h = obj['bbox']
                
                # 【关键修改】强制转换为 python int，去除 numpy 类型
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # 坐标边界保护
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x+w), min(H, y+h)
                
                if x2 > x1 and y2 > y1:
                    # 裁剪并确保是RGB
                    crop = original_image.crop((x1, y1, x2, y2))
                    if crop.mode != 'RGB':
                        crop = crop.convert('RGB')
                    crops.append(crop)
                    valid_indices.append(idx)
            
            # ... 后面的代码 ...
            
            # 初始化全0特征
            visual_feats = torch.zeros(len(objects), self.visual_dim).to(self.device)
            
            if crops:
                try:
                    img_inputs = self.clip_processor(images=crops, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        valid_visual_feats = self.clip_model.get_image_features(**img_inputs) # [M, 512]
                    # 添加 epsilon 防止除零
                    valid_visual_feats = valid_visual_feats / (valid_visual_feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                    visual_feats[valid_indices] = valid_visual_feats
                except Exception as e:
                    # Skip bad crops that PIL/processor cannot handle
                    print(f"Warning: skip invalid crop batch ({len(crops)} items): {e}")

            # --- 3. 位置特征 ---
            pos_feats = []
            for obj in objects:
                x, y, w, h = obj['bbox']
                pos = torch.tensor([x/W, y/H, w/W, h/H], device=self.device)
                pos_feats.append(pos)
            pos_feats = torch.stack(pos_feats)

            # --- 4. 融合 ---
            combined = torch.cat([visual_feats, text_feats, pos_feats], dim=-1)
            node_features = self.fusion_mlp(combined.float())
            return node_features # [N, GNN_HIDDEN_DIM]

    def __init__(self):
        super().__init__()
        # 初始化 MaskFormer
        self.processor = MaskFormerImageProcessor.from_pretrained(Config.SEGMENTATION_MODEL)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(Config.SEGMENTATION_MODEL)
        self.model.eval()
        
        # 初始化特征构建器 (这就是你漏掉的部分)
        self.feature_builder = self.NodeFeatureBuilder(device=Config.DEVICE)
    
    def forward(self, images):
        """
        执行全景分割并返回对象及特征
        Returns:
            objects_list: 对象信息列表
            features_list: 对应的GNN输入特征张量列表
        """
        # 预处理
        try:
            inputs = self.processor(images=images, return_tensors="pt")
        except Exception as e:
            print(f"Warning: skip batch in segmentation preprocessing: {e}")
            return [[] for _ in images], [torch.empty(0, Config.GNN_HIDDEN_DIM).to(Config.DEVICE) for _ in images]

        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
        
        # 分割推理（MaskFormer 推理不训练，显存更省）
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 后处理
        results = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(img.height, img.width) for img in images]
        )
        
        final_objects_list = []
        final_features_list = []
        
        for i, result in enumerate(results):
            img = images[i]
            segments_info = result['segments_info']
            # 获取 segmentation map 以计算 bbox
            panoptic_mask = result['segmentation'].cpu().numpy()
            
            objects = []
            
            # 遍历分割结果
            for segment in segments_info:
                seg_id = segment['id']
                
                # --- 关键修复：手动计算 BBox ---
                # 找到 mask 中等于当前 id 的区域
                rows, cols = np.where(panoptic_mask == seg_id)
                
                if len(rows) == 0:
                    continue # 过滤空 mask
                
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                w = x_max - x_min
                h = y_max - y_min
                
                obj = {
                    'id': segment['id'],
                    'label_id': segment['label_id'],
                    'label': self.model.config.id2label[segment['label_id']],
                    'score': segment.get('score', 1.0),
                    'area': segment.get('area', 0),
                    'bbox': [x_min, y_min, w, h] # 存入真实的 bbox
                }
                objects.append(obj)
            
            # 调用 FeatureBuilder 生成特征
            if objects:
                # 传入原图和对象列表，内部自动做 Crop 和 CLIP 编码
                node_feats = self.feature_builder(img, objects)
            else:
                node_feats = torch.empty(0, Config.GNN_HIDDEN_DIM).to(Config.DEVICE)
            
            final_objects_list.append(objects)
            final_features_list.append(node_feats)
        
        return final_objects_list, final_features_list


class RelationExtractor(nn.Module):
    """关系提取模块 - 基于规则的空间关系提取"""
    
    def __init__(self):
        super().__init__()
        # 定义空间关系类型
        self.relation_types = [
            'left_of', 'right_of', 'above', 'below',
            'near', 'far_from', 'contains', 'part_of'
        ]
        self.relation_embedding = nn.Embedding(
            len(self.relation_types), 
            Config.GNN_HIDDEN_DIM
        )

        # 关系抽取超参：尽量产生稀疏、稳定的边，避免训练标签过稠密
        self.max_relations_per_object = getattr(Config, 'REL_MAX_PER_OBJECT', 3)
        self.min_pair_score = getattr(Config, 'REL_MIN_PAIR_SCORE', 0.40)
        self.near_threshold = getattr(Config, 'REL_NEAR_THRESHOLD', 0.20)  # 相对对角线距离(0~1)，越小越严格

    def extract_spatial_relations(self, objects):
            """
            基于对象位置提取空间关系
            Args:
                objects: Batch对象列表 (List[List[Dict]])
            """
            # --- 修复逻辑 Start: 自动处理维度不一致问题 ---
            if not objects:
                return []
                
            # 检查是否传入了 Tensor (之前的错误)
            if isinstance(objects, torch.Tensor):
                raise TypeError("extract_spatial_relations 接收到了 Tensor，请检查调用处是否解包正确。")

            # 检查是否传入了单张图的对象列表 [Dict, Dict...]，而不是 Batch [[Dict...], [Dict...]]
            # 如果第一个元素是 Dict，说明缺少 Batch 维度
            if isinstance(objects, list) and len(objects) > 0 and isinstance(objects[0], dict):
                print("Warning: 检测到输入缺少 Batch 维度，正在自动修复...")
                objects = [objects] # 自动包裹一层，变成 [[Dict, Dict...]]
            # --- 修复逻辑 End ---

            # 用于存储整个 Batch 中每张图片的关系
            batch_relations = [] 
            
            # 1. 第一层循环：遍历 Batch 中的每一张图片
            for img_idx, img_objects in enumerate(objects):
                
                # 这里的 img_objects 必须是列表
                if not isinstance(img_objects, list):
                    # 再次防御，如果出现意外数据直接跳过
                    print(f"Skipping invalid image data at index {img_idx}: {type(img_objects)}")
                    batch_relations.append([])
                    continue

                current_img_relations = []
                n_objects = len(img_objects)

                if n_objects < 2:
                    batch_relations.append([])
                    continue

                # 预计算中心点与归一化距离尺度
                centers = []
                for obj in img_objects:
                    bbox = obj.get('bbox', [0, 0, 0, 0])
                    x, y, w, h = bbox
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    centers.append((x + w / 2.0, y + h / 2.0))

                xs = [c[0] for c in centers]
                ys = [c[1] for c in centers]
                # 防御：如果 bbox 全是 0，给一个非零尺度避免除0
                span_x = max(max(xs) - min(xs), 1.0)
                span_y = max(max(ys) - min(ys), 1.0)
                diag = (span_x ** 2 + span_y ** 2) ** 0.5
                diag = max(diag, 1.0)

                def _score(obj):
                    s = obj.get('score', 0.0)
                    if hasattr(s, 'item'):
                        s = s.item()
                    return float(s)

                # 为每个对象只保留最近的 top-k 邻居，避免 O(N^2) 全连接标签
                for i in range(n_objects):
                    # 计算 i 到其他点的距离
                    dists = []
                    for j in range(n_objects):
                        if i == j:
                            continue
                        dx = (centers[j][0] - centers[i][0]) / diag
                        dy = (centers[j][1] - centers[i][1]) / diag
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                        dists.append((dist, j, dx, dy))

                    dists.sort(key=lambda t: t[0])
                    dists = dists[: self.max_relations_per_object]

                    for dist, j, dx, dy in dists:
                        confidence = min(_score(img_objects[i]), _score(img_objects[j]))
                        if confidence < self.min_pair_score:
                            continue

                        # near：距离足够小
                        if dist <= self.near_threshold and 'near' in self.relation_types:
                            rel_idx = self.relation_types.index('near')
                            edge_conf = confidence * (1.0 - dist)
                            current_img_relations.append((i, rel_idx, j, float(edge_conf)))
                            continue

                        # 否则给一个方向关系（left/right/above/below）
                        if abs(dx) >= abs(dy):
                            if dx > 0 and 'right_of' in self.relation_types:
                                rel_idx = self.relation_types.index('right_of')
                            elif dx <= 0 and 'left_of' in self.relation_types:
                                rel_idx = self.relation_types.index('left_of')
                            else:
                                continue
                        else:
                            if dy > 0 and 'below' in self.relation_types:
                                rel_idx = self.relation_types.index('below')
                            elif dy <= 0 and 'above' in self.relation_types:
                                rel_idx = self.relation_types.index('above')
                            else:
                                continue

                        edge_conf = confidence * max(0.1, (1.0 - dist) * 0.5)
                        current_img_relations.append((i, rel_idx, j, float(edge_conf)))
                
                batch_relations.append(current_img_relations)
            
            return batch_relations
    
    def forward(self, objects_list):
        """
        为批次中的每张图像提取关系
        Args:
            objects_list: 对象列表的列表
        Returns:
            relations_list: 关系列表的列表
        直接传给extract_spatial_relationsbatch
        """
        
        return self.extract_spatial_relations(objects_list)


class CoreObjectSelector:
    """核心对象筛选器"""
    
    def __init__(self, min_core=Config.MIN_CORE_OBJECTS, 
                 max_core=Config.MAX_CORE_OBJECTS,
                 weights=(0.4, 0.3, 0.3)):
        self.min_core = min_core
        self.max_core = max_core
        self.weights = weights
    
    def compute_importance(self, objects, relations):
        """
        计算对象重要性分数
        Args:
            objects: 对象列表
            relations: 关系列表
        Returns:
            importance_scores: 重要性分数数组
        """
        n_objects = len(objects)
        importance = np.zeros(n_objects)
        
        w_conf, w_area, w_cent = self.weights

        # 1. 视觉显著性(基于分数和面积)
        for i, obj in enumerate(objects):
            importance[i] += obj['score'] * w_conf
            # 归一化面积
            area_score = min(obj.get('area', 0) / 10000, 1.0)
            importance[i] += area_score * w_area
        
        # 2. 关系中心性
        relation_count = np.zeros(n_objects)
        for subj_idx, _, obj_idx, conf in relations:
            relation_count[subj_idx] += conf
            relation_count[obj_idx] += conf
        
        if relation_count.max() > 1e-6:
            relation_count = relation_count / (relation_count.max() + 1e-8)
        importance += relation_count * w_cent
        
        return importance

    def _is_background_label(self, label: str) -> bool:
        """判断是否为背景/Stuff 类标签。

        说明：核心对象阶段更保守，默认把 merged/stuff/other 以及常见场景背景关键词视为背景。
        """
        label = (label or "").lower()
        if not label:
            return True

        # 经验规则：mask2former/maskformer 的 merged/stuff/other 往往是场景背景或大块材质
        if 'merged' in label or 'stuff' in label or 'other' in label:
            return True

        background_keywords = getattr(
            Config,
            'VISUAL_CONTEXT_BACKGROUND_KEYWORDS',
            (
                'wall', 'floor', 'ceiling', 'pavement', 'road', 'grass', 'sky', 'building',
                'mountain', 'dirt', 'curtain', 'rug', 'banner'
            )
        )
        return any(k in label for k in background_keywords)
    
    def select_core_objects(self, objects, relations):
        """
        选择核心对象
        Args:
            objects: 对象列表
            relations: 关系列表
        Returns:
            core_indices: 核心对象的索引列表
        """
        # 计算重要性并做全量排序（从高到低）
        importance = self.compute_importance(objects, relations)

        # 动态确定目标核心对象数量（作为 top-k 的 K），但允许最终不足 K
        n_core = min(max(self.min_core, len(objects) // 3), self.max_core)

        ranked_indices = np.argsort(importance)[::-1].tolist()

        # 顺延填充：跳过背景与重复 label，直到凑够 n_core 或遍历完
        core_indices = []
        seen_labels = set()
        for idx in ranked_indices:
            if idx < 0 or idx >= len(objects):
                continue

            obj = objects[idx]
            label = obj.get('label', '')
            if not label:
                continue

            if self._is_background_label(label):
                continue

            label_l = label.lower()
            if label_l in seen_labels:
                continue

            seen_labels.add(label_l)
            core_indices.append(idx)

            if len(core_indices) >= n_core:
                break

        # 兜底：如果全部被背景过滤/去重过滤导致为空，但确实检测到了对象，至少保留一个最重要的对象
        if not core_indices and len(objects) > 0 and ranked_indices:
            fallback_idx = ranked_indices[0]
            if 0 <= fallback_idx < len(objects):
                core_indices = [fallback_idx]

        return core_indices


class CausalGATLayer(nn.Module):
    """因果增强的图注意力层（修复版本）"""
    
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        
        # 节点特征投影
        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        
        # 边特征投影 - 修复：输出维度为num_heads
        self.edge_proj = nn.Linear(1, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index, edge_features):
        """
        Args:
            x: [N, in_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_features: [E, 1] 边特征
        """
        batch_size = x.size(0)
        
        # 投影到查询、键、值
        Q = self.Q(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.K(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.V(x).view(batch_size, self.num_heads, self.head_dim)
        
        # 计算注意力得分
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            
            # [E, num_heads, head_dim]
            Q_dst = Q[dst]
            K_src = K[src]
            V_src = V[src]
            
            # 注意力得分: [E, num_heads]
            attn_scores = (Q_dst * K_src).sum(dim=-1) / (self.head_dim ** 0.5)
            
            # 融合边特征: [E, num_heads]
            edge_attn = self.edge_proj(edge_features)  # [E, num_heads]
            attn_scores = attn_scores + edge_attn
            
            # Softmax归一化（按目标节点分组）
            attn_probs = torch.zeros_like(attn_scores)
            for i in range(batch_size):
                mask = (dst == i)
                if mask.any():
                    attn_probs[mask] = F.softmax(attn_scores[mask], dim=0)
            
            attn_probs = self.dropout(attn_probs)
            
            # 聚合消息: [E, num_heads, head_dim]
            messages = attn_probs.unsqueeze(-1) * V_src
            
            # 按目标节点聚合
            out = torch.zeros(batch_size, self.num_heads, self.head_dim, 
                            device=x.device, dtype=x.dtype)
            for i in range(batch_size):
                mask = (dst == i)
                if mask.any():
                    out[i] = messages[mask].sum(dim=0)
            
            # 重塑输出
            out = out.view(batch_size, self.out_dim)
        else:
            # 没有边时，直接使用值向量
            out = V.view(batch_size, self.out_dim)
        
        # 残差连接和归一化
        if x.size(-1) == self.out_dim:
            out = self.norm(out + x)
        else:
            out = self.norm(out)
        
        return out


class CausalGNN(nn.Module):
    """因果增强的图神经网络（修复版本）"""
    
    def __init__(self, hidden_dim=Config.GNN_HIDDEN_DIM, 
                 num_layers=Config.GNN_NUM_LAYERS,
                 num_heads=Config.GNN_NUM_HEADS,
                 dropout=Config.GNN_DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GAT层
        self.gat_layers = nn.ModuleList([
            CausalGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 共现和因果预测头
        self.cooccurrence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.causality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, edge_index, edge_features):
        """
        Args:
            node_features: [N, hidden_dim]
            edge_index: [2, E]
            edge_features: [E, 1]
        Returns:
            node_embeddings: [N, hidden_dim]
            cooccurrence_matrix: [N, N]
            causality_matrix: [N, N]
        """
        h = node_features
        
        # 通过GAT层
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_features)
        
        # 构建成对特征 [N, N, 2*hidden_dim]
        n = h.size(0)
        h_i = h.unsqueeze(1).expand(n, n, -1)  # [N, N, hidden_dim]
        h_j = h.unsqueeze(0).expand(n, n, -1)  # [N, N, hidden_dim]
        pair_features = torch.cat([h_i, h_j], dim=-1)  # [N, N, 2*hidden_dim]
        
        # 预测共现和因果关系
        cooccurrence = self.cooccurrence_head(pair_features).squeeze(-1)  # [N, N]
        causality = self.causality_head(pair_features).squeeze(-1)  # [N, N]
        
        return h, cooccurrence, causality


class UncertaintyEstimator:
    """不确定性推理模块"""
    
    def __init__(self, num_samples=Config.MC_DROPOUT_SAMPLES):
        self.num_samples = num_samples
    
    def mc_dropout_inference(self, model, node_features, edge_index, edge_features):
        """
        使用Monte Carlo Dropout进行不确定性估计
        Args:
            model: GNN模型
            node_features, edge_index, edge_features: 图数据
        Returns:
            mean_probs: 平均预测概率
            uncertainty: 不确定性分数
        """
        model.train()  # 启用dropout
        
        all_cooccurrence = []
        all_causality = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                _, cooccur, causal = model(node_features, edge_index, edge_features)
                all_cooccurrence.append(cooccur)
                all_causality.append(causal)
        
        model.eval()
        
        # 计算均值和标准差
        cooccur_tensor = torch.stack(all_cooccurrence)  # [num_samples, N, N]
        causal_tensor = torch.stack(all_causality)
        
        mean_cooccur = cooccur_tensor.mean(dim=0)
        std_cooccur = cooccur_tensor.std(dim=0)
        
        mean_causal = causal_tensor.mean(dim=0)
        std_causal = causal_tensor.std(dim=0)
        
        # 不确定性 = 标准差
        uncertainty_cooccur = std_cooccur
        uncertainty_causal = std_causal
        
        return {
            'cooccurrence': mean_cooccur,
            'causality': mean_causal,
            'uncertainty_cooccur': uncertainty_cooccur,
            'uncertainty_causal': uncertainty_causal
        }
    
    def compute_existence_probability(self, core_objects, all_objects, 
                                       cooccurrence_matrix, causality_matrix):
        """
        计算对象存在概率
        Args:
            core_objects: 核心对象索引
            all_objects: 所有对象
            cooccurrence_matrix: [N, N] 共现概率矩阵
            causality_matrix: [N, N] 因果强度矩阵
        Returns:
            existence_probs: 每个对象的存在概率
        """
        N = len(all_objects)
        existence_probs = torch.zeros(N, device=cooccurrence_matrix.device)

        # 防御性处理：极端情况下核心对象为空时，直接返回全 0（避免除零）
        if not core_objects:
            return existence_probs
        
        for x in range(N):
            if x in core_objects:
                existence_probs[x] = 1.0
            else:
                # P(ox) = σ(Σ wi,x * P(ox|oi) + Σ wx,j * P(oj|ox))
                incoming_score = 0
                outgoing_score = 0
                
                for i in core_objects:
                    incoming_score += causality_matrix[i, x] * cooccurrence_matrix[i, x]
                    outgoing_score += causality_matrix[x, i] * cooccurrence_matrix[x, i]
                
                # 注意：cooccurrence/causality 已经是 Sigmoid 输出(0~1)，这里再 Sigmoid 会导致无证据时也≈0.5
                # 我们把平均乘积当作“存在概率”的近似，直接裁剪到[0,1]
                denom = max(len(core_objects) * 2, 1)
                total_score = (incoming_score + outgoing_score) / (denom + 1e-8)
                existence_probs[x] = total_score.clamp(0.0, 1.0)
        
        return existence_probs


if __name__ == "__main__":
    # 测试模块
    Config.create_dirs()
    device = Config.DEVICE
    
    print("测试全景分割模块...")
    from PIL import Image
    import glob
    import os
    
    # 使用本地图像进行测试
    val_images = glob.glob('/root/autodl-tmp/HuL_Code/vision-caption-correction/data/coco/val2017/*.jpg')
    if val_images:
        test_image = Image.open(val_images[0])
        
        seg_module = PanopticSegmentationModule().to(device)
        objects_list = seg_module([test_image])
        print(f"✓ 检测到 {len(objects_list[0])} 个对象")
        
        print("[DEBUG] Detected objects:")
        for obj in objects_list[0]:
            print(f"  Label: {obj['label']}, Score: {obj['score']}, BBox: {obj['bbox']}")

        print("\n测试关系提取模块...")
        rel_extractor = RelationExtractor().to(device)
        relations_list = rel_extractor(objects_list)
        print(f"✓ 提取了 {len(relations_list[0])} 个关系")
        
        print("[DEBUG] Extracted relations:")
        for rel in relations_list[0]:
            subj_idx, rel_type, obj_idx, confidence = rel
            print(f"  Relation: {rel_extractor.relation_types[rel_type]} between Object {subj_idx} and Object {obj_idx} with Confidence: {confidence}")

        print("\n测试核心对象选择...")
        selector = CoreObjectSelector()
        core_indices = selector.select_core_objects(objects_list[0], relations_list[0])
        print(f"✓ 选择了 {len(core_indices)} 个核心对象")
        
        print("\n测试GNN模型...")
        # 构建图数据
        n_objects = len(objects_list[0])
        node_features = torch.randn(n_objects, Config.GNN_HIDDEN_DIM).to(device)
        
        # 构建边
        edges = []
        edge_feats = []
        for subj, rel, obj, conf in relations_list[0]:
            edges.append([subj, obj])
            edge_feats.append([conf])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
            edge_features = torch.tensor(edge_feats, dtype=torch.float).to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            edge_features = torch.empty((0, 1), dtype=torch.float).to(device)
        
        gnn = CausalGNN().to(device)
        embeddings, cooccur, causal = gnn(node_features, edge_index, edge_features)
        print(f"✓ GNN输出维度: {embeddings.shape}")
        print(f"✓ 共现矩阵维度: {cooccur.shape}")
        print(f"✓ 因果矩阵维度: {causal.shape}")
        
        print("\n✓ 所有模块测试通过!")
    else:
        print("✗ 找不到测试图像，请确保数据集已下载")
