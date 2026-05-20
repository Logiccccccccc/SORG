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
import torch.nn.functional as F
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import cdist
import math
from config import Config

class PanopticSegmentationModule(nn.Module):

    class NodeFeatureBuilder(nn.Module):

        def __init__(self, device=Config.DEVICE):
            super().__init__()
            self.device = device
            print('Loading CLIP for Feature Extraction...')
            self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
            self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.visual_dim = 512
            self.text_dim = 512
            self.pos_dim = 4
            self.out_dim = Config.GNN_HIDDEN_DIM
            self.fusion_mlp = nn.Sequential(nn.Linear(self.visual_dim + self.text_dim + self.pos_dim, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, self.out_dim), nn.LayerNorm(self.out_dim))

        def forward(self, original_image, objects):
            if not objects:
                return torch.empty(0, self.out_dim).to(self.device)
            labels = [obj['label'] for obj in objects]
            text_inputs = self.clip_processor(text=labels, return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad():
                text_feats = self.clip_model.get_text_features(**text_inputs)
            text_feats = text_feats / (text_feats.norm(p=2, dim=-1, keepdim=True) + 1e-08)
            crops = []
            valid_indices = []
            W, H = original_image.size
            W = max(W, 1)
            H = max(H, 1)
            for idx, obj in enumerate(objects):
                x, y, w, h = obj['bbox']
                x, y, w, h = (int(x), int(y), int(w), int(h))
                x1, y1 = (max(0, x), max(0, y))
                x2, y2 = (min(W, x + w), min(H, y + h))
                if x2 > x1 and y2 > y1:
                    crop = original_image.crop((x1, y1, x2, y2))
                    if crop.mode != 'RGB':
                        crop = crop.convert('RGB')
                    crops.append(crop)
                    valid_indices.append(idx)
            visual_feats = torch.zeros(len(objects), self.visual_dim).to(self.device)
            if crops:
                try:
                    img_inputs = self.clip_processor(images=crops, return_tensors='pt').to(self.device)
                    with torch.no_grad():
                        valid_visual_feats = self.clip_model.get_image_features(**img_inputs)
                    valid_visual_feats = valid_visual_feats / (valid_visual_feats.norm(p=2, dim=-1, keepdim=True) + 1e-08)
                    visual_feats[valid_indices] = valid_visual_feats
                except Exception as e:
                    print(f'Warning: skip invalid crop batch ({len(crops)} items): {e}')
            pos_feats = []
            for obj in objects:
                x, y, w, h = obj['bbox']
                pos = torch.tensor([x / W, y / H, w / W, h / H], device=self.device)
                pos_feats.append(pos)
            pos_feats = torch.stack(pos_feats)
            combined = torch.cat([visual_feats, text_feats, pos_feats], dim=-1)
            node_features = self.fusion_mlp(combined.float())
            return node_features

    def __init__(self):
        super().__init__()
        self.processor = MaskFormerImageProcessor.from_pretrained(Config.SEGMENTATION_MODEL)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(Config.SEGMENTATION_MODEL)
        self.model.eval()
        self.feature_builder = self.NodeFeatureBuilder(device=Config.DEVICE)

    def forward(self, images):
        try:
            inputs = self.processor(images=images, return_tensors='pt')
        except Exception as e:
            print(f'Warning: skip batch in segmentation preprocessing: {e}')
            return ([[] for _ in images], [torch.empty(0, Config.GNN_HIDDEN_DIM).to(Config.DEVICE) for _ in images])
        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[(img.height, img.width) for img in images])
        final_objects_list = []
        final_features_list = []
        for i, result in enumerate(results):
            img = images[i]
            segments_info = result['segments_info']
            panoptic_mask = result['segmentation'].cpu().numpy()
            objects = []
            for segment in segments_info:
                seg_id = segment['id']
                rows, cols = np.where(panoptic_mask == seg_id)
                if len(rows) == 0:
                    continue
                y_min, y_max = (rows.min(), rows.max())
                x_min, x_max = (cols.min(), cols.max())
                w = x_max - x_min
                h = y_max - y_min
                obj = {'id': segment['id'], 'label_id': segment['label_id'], 'label': self.model.config.id2label[segment['label_id']], 'score': segment.get('score', 1.0), 'area': segment.get('area', 0), 'bbox': [x_min, y_min, w, h]}
                objects.append(obj)
            if objects:
                node_feats = self.feature_builder(img, objects)
            else:
                node_feats = torch.empty(0, Config.GNN_HIDDEN_DIM).to(Config.DEVICE)
            final_objects_list.append(objects)
            final_features_list.append(node_feats)
        return (final_objects_list, final_features_list)

class RelationExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.relation_types = ['left_of', 'right_of', 'above', 'below', 'near', 'far_from', 'contains', 'part_of']
        self.relation_embedding = nn.Embedding(len(self.relation_types), Config.GNN_HIDDEN_DIM)
        self.max_relations_per_object = getattr(Config, 'REL_MAX_PER_OBJECT', 3)
        self.min_pair_score = getattr(Config, 'REL_MIN_PAIR_SCORE', 0.4)
        self.near_threshold = getattr(Config, 'REL_NEAR_THRESHOLD', 0.2)

    def extract_spatial_relations(self, objects):
        if not objects:
            return []
        if isinstance(objects, torch.Tensor):
            raise TypeError('extract_spatial_relations received a Tensor. Check whether the caller unpacked the inputs correctly.')
        if isinstance(objects, list) and len(objects) > 0 and isinstance(objects[0], dict):
            print('Warning: input is missing the batch dimension. It will be wrapped automatically.')
            objects = [objects]
        batch_relations = []
        for img_idx, img_objects in enumerate(objects):
            if not isinstance(img_objects, list):
                print(f'Skipping invalid image data at index {img_idx}: {type(img_objects)}')
                batch_relations.append([])
                continue
            current_img_relations = []
            n_objects = len(img_objects)
            if n_objects < 2:
                batch_relations.append([])
                continue
            centers = []
            for obj in img_objects:
                bbox = obj.get('bbox', [0, 0, 0, 0])
                x, y, w, h = bbox
                x, y, w, h = (float(x), float(y), float(w), float(h))
                centers.append((x + w / 2.0, y + h / 2.0))
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            span_x = max(max(xs) - min(xs), 1.0)
            span_y = max(max(ys) - min(ys), 1.0)
            diag = (span_x ** 2 + span_y ** 2) ** 0.5
            diag = max(diag, 1.0)

            def _score(obj):
                s = obj.get('score', 0.0)
                if hasattr(s, 'item'):
                    s = s.item()
                return float(s)
            for i in range(n_objects):
                dists = []
                for j in range(n_objects):
                    if i == j:
                        continue
                    dx = (centers[j][0] - centers[i][0]) / diag
                    dy = (centers[j][1] - centers[i][1]) / diag
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    dists.append((dist, j, dx, dy))
                dists.sort(key=lambda t: t[0])
                dists = dists[:self.max_relations_per_object]
                for dist, j, dx, dy in dists:
                    confidence = min(_score(img_objects[i]), _score(img_objects[j]))
                    if confidence < self.min_pair_score:
                        continue
                    if dist <= self.near_threshold and 'near' in self.relation_types:
                        rel_idx = self.relation_types.index('near')
                        edge_conf = confidence * (1.0 - dist)
                        current_img_relations.append((i, rel_idx, j, float(edge_conf)))
                        continue
                    if abs(dx) >= abs(dy):
                        if dx > 0 and 'right_of' in self.relation_types:
                            rel_idx = self.relation_types.index('right_of')
                        elif dx <= 0 and 'left_of' in self.relation_types:
                            rel_idx = self.relation_types.index('left_of')
                        else:
                            continue
                    elif dy > 0 and 'below' in self.relation_types:
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
        return self.extract_spatial_relations(objects_list)

class CoreObjectSelector:

    def __init__(self, min_core=Config.MIN_CORE_OBJECTS, max_core=Config.MAX_CORE_OBJECTS, weights=(0.4, 0.3, 0.3)):
        self.min_core = min_core
        self.max_core = max_core
        self.weights = weights

    def compute_importance(self, objects, relations):
        n_objects = len(objects)
        importance = np.zeros(n_objects)
        w_conf, w_area, w_cent = self.weights
        for i, obj in enumerate(objects):
            importance[i] += obj['score'] * w_conf
            area_score = min(obj.get('area', 0) / 10000, 1.0)
            importance[i] += area_score * w_area
        relation_count = np.zeros(n_objects)
        for subj_idx, _, obj_idx, conf in relations:
            relation_count[subj_idx] += conf
            relation_count[obj_idx] += conf
        if relation_count.max() > 1e-06:
            relation_count = relation_count / (relation_count.max() + 1e-08)
        importance += relation_count * w_cent
        return importance

    def _is_background_label(self, label: str) -> bool:
        label = (label or '').lower()
        if not label:
            return True
        if 'merged' in label or 'stuff' in label or 'other' in label:
            return True
        background_keywords = getattr(Config, 'VISUAL_CONTEXT_BACKGROUND_KEYWORDS', ('wall', 'floor', 'ceiling', 'pavement', 'road', 'grass', 'sky', 'building', 'mountain', 'dirt', 'curtain', 'rug', 'banner'))
        return any((k in label for k in background_keywords))

    def select_core_objects(self, objects, relations):
        importance = self.compute_importance(objects, relations)
        n_core = min(max(self.min_core, len(objects) // 3), self.max_core)
        ranked_indices = np.argsort(importance)[::-1].tolist()
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
        if not core_indices and len(objects) > 0 and ranked_indices:
            fallback_idx = ranked_indices[0]
            if 0 <= fallback_idx < len(objects):
                core_indices = [fallback_idx]
        return core_indices

class CausalGATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(1, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_features):
        batch_size = x.size(0)
        Q = self.Q(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.K(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.V(x).view(batch_size, self.num_heads, self.head_dim)
        if edge_index.size(1) > 0:
            src, dst = (edge_index[0], edge_index[1])
            Q_dst = Q[dst]
            K_src = K[src]
            V_src = V[src]
            attn_scores = (Q_dst * K_src).sum(dim=-1) / self.head_dim ** 0.5
            edge_attn = self.edge_proj(edge_features)
            attn_scores = attn_scores + edge_attn
            attn_probs = torch.zeros_like(attn_scores)
            for i in range(batch_size):
                mask = dst == i
                if mask.any():
                    attn_probs[mask] = F.softmax(attn_scores[mask], dim=0)
            attn_probs = self.dropout(attn_probs)
            messages = attn_probs.unsqueeze(-1) * V_src
            out = torch.zeros(batch_size, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
            for i in range(batch_size):
                mask = dst == i
                if mask.any():
                    out[i] = messages[mask].sum(dim=0)
            out = out.view(batch_size, self.out_dim)
        else:
            out = V.view(batch_size, self.out_dim)
        if x.size(-1) == self.out_dim:
            out = self.norm(out + x)
        else:
            out = self.norm(out)
        return out

class CausalGNN(nn.Module):

    def __init__(self, hidden_dim=Config.GNN_HIDDEN_DIM, num_layers=Config.GNN_NUM_LAYERS, num_heads=Config.GNN_NUM_HEADS, dropout=Config.GNN_DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList([CausalGATLayer(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.cooccurrence_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.causality_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, node_features, edge_index, edge_features):
        h = node_features
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_features)
        n = h.size(0)
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        pair_features = torch.cat([h_i, h_j], dim=-1)
        cooccurrence = self.cooccurrence_head(pair_features).squeeze(-1)
        causality = self.causality_head(pair_features).squeeze(-1)
        return (h, cooccurrence, causality)

class UncertaintyEstimator:

    def __init__(self, num_samples=Config.MC_DROPOUT_SAMPLES):
        self.num_samples = num_samples

    def mc_dropout_inference(self, model, node_features, edge_index, edge_features):
        model.train()
        all_cooccurrence = []
        all_causality = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                _, cooccur, causal = model(node_features, edge_index, edge_features)
                all_cooccurrence.append(cooccur)
                all_causality.append(causal)
        model.eval()
        cooccur_tensor = torch.stack(all_cooccurrence)
        causal_tensor = torch.stack(all_causality)
        mean_cooccur = cooccur_tensor.mean(dim=0)
        std_cooccur = cooccur_tensor.std(dim=0)
        mean_causal = causal_tensor.mean(dim=0)
        std_causal = causal_tensor.std(dim=0)
        uncertainty_cooccur = std_cooccur
        uncertainty_causal = std_causal
        return {'cooccurrence': mean_cooccur, 'causality': mean_causal, 'uncertainty_cooccur': uncertainty_cooccur, 'uncertainty_causal': uncertainty_causal}

    def compute_existence_probability(self, core_objects, all_objects, cooccurrence_matrix, causality_matrix):
        N = len(all_objects)
        existence_probs = torch.zeros(N, device=cooccurrence_matrix.device)
        if not core_objects:
            return existence_probs
        for x in range(N):
            if x in core_objects:
                existence_probs[x] = 1.0
            else:
                incoming_score = 0
                outgoing_score = 0
                for i in core_objects:
                    incoming_score += causality_matrix[i, x] * cooccurrence_matrix[i, x]
                    outgoing_score += causality_matrix[x, i] * cooccurrence_matrix[x, i]
                denom = max(len(core_objects) * 2, 1)
                total_score = (incoming_score + outgoing_score) / (denom + 1e-08)
                existence_probs[x] = total_score.clamp(0.0, 1.0)
        return existence_probs
if __name__ == '__main__':
    Config.create_dirs()
    device = Config.DEVICE
    print('Testing panoptic segmentation module...')
    from PIL import Image
    import glob
    import os
    val_images = glob.glob(os.path.join(Config.COCO_VAL_IMG, '*.jpg'))
    if val_images:
        test_image = Image.open(val_images[0])
        seg_module = PanopticSegmentationModule().to(device)
        objects_list, features_list = seg_module([test_image])
        print(f'Detected {len(objects_list[0])} objects')
        print('[DEBUG] Detected objects:')
        for obj in objects_list[0]:
            print(f"  Label: {obj['label']}, Score: {obj['score']}, BBox: {obj['bbox']}")
        print('\nTesting relation extraction module...')
        rel_extractor = RelationExtractor().to(device)
        relations_list = rel_extractor(objects_list)
        print(f'Extracted {len(relations_list[0])} relations')
        print('[DEBUG] Extracted relations:')
        for rel in relations_list[0]:
            subj_idx, rel_type, obj_idx, confidence = rel
            print(f'  Relation: {rel_extractor.relation_types[rel_type]} between Object {subj_idx} and Object {obj_idx} with Confidence: {confidence}')
        print('\nTesting core object selection...')
        selector = CoreObjectSelector()
        core_indices = selector.select_core_objects(objects_list[0], relations_list[0])
        print(f'Selected {len(core_indices)} core objects')
        print('\nTesting GNN model...')
        n_objects = len(objects_list[0])
        node_features = torch.randn(n_objects, Config.GNN_HIDDEN_DIM).to(device)
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
        print(f'GNN output shape: {embeddings.shape}')
        print(f'Cooccurrence matrix shape: {cooccur.shape}')
        print(f'Causality matrix shape: {causal.shape}')
        print('\nAll module tests passed.')
    else:
        print('No test image was found. Please make sure the dataset is available.')
