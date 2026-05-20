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
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
import json
from PIL import Image
from config import Config
from model_core import PanopticSegmentationModule, RelationExtractor, CoreObjectSelector, CausalGNN

class KnowledgeBaseBuilder:

    def __init__(self, train_image_paths=None, val_image_paths=None, coco_instances_file=None, coco_captions_file=None, **kwargs):
        self.device = Config.DEVICE
        self.train_image_paths = train_image_paths
        self.val_image_paths = val_image_paths
        self.coco_instances_file = coco_instances_file
        self.coco_captions_file = coco_captions_file
        self.extra_kwargs = kwargs
        self.segmentation = PanopticSegmentationModule().to(self.device)
        self.relation_extractor = RelationExtractor().to(self.device)
        self.core_selector = CoreObjectSelector()
        self.gnn = CausalGNN().to(self.device)
        self.object_vocab = []
        self.vocab_to_idx = {}
        self.global_cooccurrence = defaultdict(lambda: defaultdict(float))
        self.causality_pairs = defaultdict(lambda: defaultdict(float))
        self.cooccurrence_matrix = None
        self.causality_matrix = None

    def build_vocabulary(self, data_loader):
        print('Building object vocabulary...')
        object_counter = defaultdict(int)
        for batch in tqdm(data_loader, desc='Scanning objects'):
            print(f'DEBUG: Batch keys available: {batch.keys()}')
            images = batch['images']
            objects_list, _ = self.segmentation(images)
            for objects in objects_list:
                for obj in objects:
                    object_counter[obj['label']] += 1
        sorted_objects = sorted(object_counter.items(), key=lambda x: x[1], reverse=True)
        self.object_vocab = [obj for obj, _ in sorted_objects]
        self.vocab_to_idx = {obj: idx for idx, obj in enumerate(self.object_vocab)}
        print(f'Vocabulary size: {len(self.object_vocab)}')

    def extract_cooccurrence_patterns(self, data_loader):
        print('Extracting co-occurrence patterns...')
        for batch in tqdm(data_loader, desc='Analyzing co-occurrence'):
            images = batch['images']
            objects_list, _ = self.segmentation(images)
            relations_list = self.relation_extractor(objects_list)
            for objects, relations in zip(objects_list, relations_list):
                object_labels = [obj['label'] for obj in objects if obj['label'] in self.vocab_to_idx]
                for i, obj1 in enumerate(object_labels):
                    for obj2 in object_labels[i + 1:]:
                        self.global_cooccurrence[obj1][obj2] += 1
                        self.global_cooccurrence[obj2][obj1] += 1
                for subj_idx, _, obj_idx, conf in relations:
                    if subj_idx >= len(objects) or obj_idx >= len(objects):
                        continue
                    label_subj = objects[subj_idx]['label']
                    label_obj = objects[obj_idx]['label']
                    if label_subj not in self.vocab_to_idx or label_obj not in self.vocab_to_idx:
                        continue
                    self.causality_pairs[label_subj][label_obj] += float(conf)

    def train_gnn(self, data_loader, num_epochs=Config.NUM_EPOCHS, resume_path=None, disable_dropout: bool=False, target_mode: str='binary'):
        print('Starting GNN training...')
        params_to_optimize = list(self.gnn.parameters()) + list(self.segmentation.feature_builder.parameters())
        optimizer = optim.AdamW(params_to_optimize, lr=Config.GNN_LEARNING_RATE, weight_decay=Config.GNN_WEIGHT_DECAY)
        if disable_dropout:
            self.segmentation.feature_builder.eval()
            self.gnn.eval()
        else:
            self.segmentation.feature_builder.train()
            self.gnn.train()
        start_epoch = 0
        if resume_path is not None and os.path.isfile(resume_path):
            print(f'Resuming from checkpoint: {resume_path}')
            checkpoint = torch.load(resume_path, map_location=self.device)
            if 'gnn_state_dict' in checkpoint:
                self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f'Resumed from epoch {start_epoch}')
            else:
                self.gnn.load_state_dict(checkpoint)
                print('Only model parameters were restored; optimizer and epoch information were not found.')
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            num_batches = 0
            progress_bar = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in progress_bar:
                images = batch['images']
                objects_list, features_list = self.segmentation(images)
                relations_list = self.relation_extractor(objects_list)
                batch_loss = 0
                valid_samples = 0
                for objects, relations, features in zip(objects_list, relations_list, features_list):
                    if len(objects) < 2:
                        continue
                    n_objects = len(objects)
                    node_features = features
                    if node_features.size(0) == 0:
                        node_features = torch.zeros(n_objects, Config.GNN_HIDDEN_DIM).to(self.device)
                    edges = []
                    edge_feats = []
                    for subj, rel, obj, conf in relations:
                        edges.append([subj, obj])
                        edge_feats.append([conf])
                    if not edges:
                        continue
                    edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
                    edge_features = torch.tensor(edge_feats, dtype=torch.float).to(self.device)
                    _, cooccur, causal = self.gnn(node_features, edge_index, edge_features)
                    target_cooccur = torch.zeros(n_objects, n_objects).to(self.device)
                    target_causal = torch.zeros(n_objects, n_objects).to(self.device)
                    relation_set = {(int(subj), int(obj)) for subj, _, obj, _ in relations}
                    if target_mode == 'binary':
                        denom = max(n_objects * (n_objects - 1), 1)
                        density = len(relation_set) / denom
                        if density > 0.5:
                            print(f'Warning: relation labels are too dense (density={density:.2f}, edges={len(relation_set)}, N={n_objects}),Binary training may show false convergence. Consider improving relation extraction or using --target_mode soft.')
                    for i, obj_i in enumerate(objects):
                        for j, obj_j in enumerate(objects):
                            if i == j:
                                continue
                            if target_mode == 'binary':
                                target_cooccur[i, j] = 1.0 if (i, j) in relation_set else 0.0
                                target_causal[i, j] = 1.0 if (i, j) in relation_set else 0.0
                            else:
                                label_i = obj_i['label']
                                label_j = obj_j['label']
                                cooccur_count = 0.0
                                if label_i in self.global_cooccurrence and label_j in self.global_cooccurrence[label_i]:
                                    cooccur_count = float(self.global_cooccurrence[label_i][label_j])
                                causal_count = 0.0
                                if label_i in self.causality_pairs and label_j in self.causality_pairs[label_i]:
                                    causal_count = float(self.causality_pairs[label_i][label_j])
                                target_cooccur[i, j] = min(cooccur_count / 100.0, 1.0) if cooccur_count > 0 else 0.0
                                target_causal[i, j] = min(causal_count / 50.0, 1.0) if causal_count > 0 else 0.0
                    loss_cooccur = F.binary_cross_entropy(cooccur, target_cooccur)
                    loss_causal = F.binary_cross_entropy(causal, target_causal)
                    loss = loss_cooccur + loss_causal
                    batch_loss += loss
                    valid_samples += 1
                if valid_samples > 0:
                    avg_loss = batch_loss / valid_samples
                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()
                    epoch_loss += avg_loss.item()
                    num_batches += 1
                    progress_bar.set_postfix({'loss': f'{avg_loss.item():.4f}'})
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                print(f'Epoch {epoch + 1}/{num_epochs}, average loss: {avg_epoch_loss:.4f}')
            if (epoch + 1) % 10 == 0:
                ckpt_dir = Config.CHECKPOINT_DIR
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f'new_gnn_epoch_{epoch + 1}.pt')
                torch.save({'gnn_state_dict': self.gnn.state_dict(), 'feature_builder_state_dict': self.segmentation.feature_builder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}, ckpt_path)
                print(f'Saved GNN weights: {ckpt_path}')

    def aggregate_knowledge(self, data_loader):
        print('Aggregating knowledge base...')
        vocab_size = len(self.object_vocab)
        device = self.device
        self.gnn.eval()
        self.segmentation.feature_builder.eval()
        self.cooccurrence_matrix = torch.zeros(vocab_size, vocab_size, device=device)
        self.causality_matrix = torch.zeros(vocab_size, vocab_size, device=device)
        pair_counts = torch.zeros(vocab_size, vocab_size, device=device)
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Aggregating inference'):
                images = batch['images']
                objects_list, features_list = self.segmentation(images)
                relations_list = self.relation_extractor(objects_list)
                for objects, relations, features in zip(objects_list, relations_list, features_list):
                    if len(objects) < 2:
                        continue
                    edges = []
                    edge_feats = []
                    for subj, rel_idx, obj, conf in relations:
                        edges.append([subj, obj])
                        edge_feats.append([conf])
                    if not edges:
                        continue
                    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
                    edge_features = torch.tensor(edge_feats, dtype=torch.float, device=device)
                    node_features = features
                    if node_features.size(0) == 0:
                        continue
                    _, cooccur, causal = self.gnn(node_features, edge_index, edge_features)
                    unknown_label_count = 0
                    for local_i, obj_i in enumerate(objects):
                        label_i = obj_i['label']
                        if label_i not in self.vocab_to_idx:
                            unknown_label_count += 1
                            continue
                        global_i = self.vocab_to_idx[label_i]
                        for local_j, obj_j in enumerate(objects):
                            if local_i == local_j:
                                continue
                            label_j = obj_j['label']
                            if label_j not in self.vocab_to_idx:
                                unknown_label_count += 1
                                continue
                            global_j = self.vocab_to_idx[label_j]
                            self.cooccurrence_matrix[global_i, global_j] += cooccur[local_i, local_j]
                            self.causality_matrix[global_i, global_j] += causal[local_i, local_j]
                            pair_counts[global_i, global_j] += 1
                    if unknown_label_count > 0:
                        print(f'Warning: aggregation skipped {unknown_label_count} unknown labels that were not in the vocabulary.')
        mask = pair_counts > 0
        if mask.any():
            self.cooccurrence_matrix[mask] /= pair_counts[mask]
            self.causality_matrix[mask] /= pair_counts[mask]
        self.cooccurrence_matrix = self.cooccurrence_matrix.cpu()
        self.causality_matrix = self.causality_matrix.cpu()
        print('Knowledge base aggregation completed.')

    def save_knowledge_base(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        vocab_data = {'vocab': self.object_vocab, 'vocab_to_idx': self.vocab_to_idx}
        with open(os.path.join(save_path, 'object_vocab.json'), 'w') as f:
            json.dump(vocab_data, f, indent=2)
        stats = {'global_cooccurrence': {k: dict(v) for k, v in self.global_cooccurrence.items()}, 'causality_pairs': {k: dict(v) for k, v in self.causality_pairs.items()}}
        with open(os.path.join(save_path, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        if self.cooccurrence_matrix is not None:
            np.save(os.path.join(save_path, 'cooccurrence_matrix.npy'), self.cooccurrence_matrix.cpu().numpy())
        if self.causality_matrix is not None:
            np.save(os.path.join(save_path, 'causality_matrix.npy'), self.causality_matrix.cpu().numpy())
        print(f'Knowledge base saved to: {save_path}')

    def load_knowledge_base(self, kb_path):
        print(f'Loading knowledge base from: {kb_path}')
        vocab_path = os.path.join(kb_path, 'object_vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                self.object_vocab = vocab_data['vocab']
                self.vocab_to_idx = vocab_data['vocab_to_idx']
            print(f'Loaded vocabulary: {len(self.object_vocab)} objects')
        else:
            print(f'Vocabulary file does not exist: {vocab_path}')
        stats_path = os.path.join(kb_path, 'statistics.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.global_cooccurrence = defaultdict(lambda: defaultdict(float))
                self.causality_pairs = defaultdict(lambda: defaultdict(float))
                for obj1, obj2_dict in stats.get('global_cooccurrence', {}).items():
                    for obj2, count in obj2_dict.items():
                        self.global_cooccurrence[obj1][obj2] = count
                for obj1, obj2_dict in stats.get('causality_pairs', {}).items():
                    for obj2, count in obj2_dict.items():
                        self.causality_pairs[obj1][obj2] = count
            print(f'Loaded statistics')
        else:
            print(f'Statistics file does not exist: {stats_path}')
        cooccur_path = os.path.join(kb_path, 'cooccurrence_matrix.npy')
        if os.path.exists(cooccur_path):
            self.cooccurrence_matrix = torch.from_numpy(np.load(cooccur_path)).to(Config.DEVICE)
            print(f'Loaded co-occurrence matrix: {self.cooccurrence_matrix.shape}')
        else:
            print(f'Co-occurrence matrix file does not exist: {cooccur_path}')
        causal_path = os.path.join(kb_path, 'causality_matrix.npy')
        if os.path.exists(causal_path):
            self.causality_matrix = torch.from_numpy(np.load(causal_path)).to(Config.DEVICE)
            print(f'Loaded causality matrix: {self.causality_matrix.shape}')
        else:
            print(f'Causality matrix file does not exist: {causal_path}')
        print('Knowledge base loading completed.')

    def build(self, data_loader, num_epochs=Config.NUM_EPOCHS, resume_path=None, disable_dropout: bool=False, target_mode: str='binary'):
        self.build_vocabulary(data_loader)
        self.extract_cooccurrence_patterns(data_loader)
        self.train_gnn(data_loader, num_epochs, resume_path=resume_path, disable_dropout=disable_dropout, target_mode=target_mode)
        self.aggregate_knowledge(data_loader)
        print('Knowledge base build completed.')
if __name__ == '__main__':
    print('Use train.py to build the knowledge base and train the GNN.')
