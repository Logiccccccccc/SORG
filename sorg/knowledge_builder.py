"""知识库构建模块"""
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
from model_core import (
    PanopticSegmentationModule, 
    RelationExtractor, 
    CoreObjectSelector,
    CausalGNN
)


class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(
        self,
        train_image_paths=None,
        val_image_paths=None,
        coco_instances_file=None,
        coco_captions_file=None,
        **kwargs,
    ):
        self.device = Config.DEVICE

        # 兼容 overfit / 评估脚本的可选输入（当前构建流程不会强依赖这些字段）
        self.train_image_paths = train_image_paths
        self.val_image_paths = val_image_paths
        self.coco_instances_file = coco_instances_file
        self.coco_captions_file = coco_captions_file
        self.extra_kwargs = kwargs
        
        # 分割和关系提取模块
        self.segmentation = PanopticSegmentationModule().to(self.device)
        self.relation_extractor = RelationExtractor().to(self.device)
        self.core_selector = CoreObjectSelector()
        
        # GNN模型
        self.gnn = CausalGNN().to(self.device)
        
        # 全局对象词汇表
        self.object_vocab = []
        self.vocab_to_idx = {}
        
        # 全局统计
        self.global_cooccurrence = defaultdict(lambda: defaultdict(float))
        self.causality_pairs = defaultdict(lambda: defaultdict(float))
        
        # 学习到的矩阵
        self.cooccurrence_matrix = None
        self.causality_matrix = None
    
    def build_vocabulary(self, data_loader):
        """
        构建对象词汇表
        Args:
            data_loader: 数据加载器
        """
        print("构建对象词汇表...")
        object_counter = defaultdict(int)
        
        for batch in tqdm(data_loader, desc="扫描对象"):
            print(f"DEBUG: Batch keys available: {batch.keys()}")
            images = batch['images']
            
            # 提取对象
            objects_list, _ = self.segmentation(images)
            
            for objects in objects_list:
                for obj in objects:
                    object_counter[obj['label']] += 1
        
        # 按频率排序，构建词汇表
        sorted_objects = sorted(object_counter.items(), key=lambda x: x[1], reverse=True)
        self.object_vocab = [obj for obj, _ in sorted_objects]
        self.vocab_to_idx = {obj: idx for idx, obj in enumerate(self.object_vocab)}
        
        print(f"词汇表大小: {len(self.object_vocab)}")
    
    def extract_cooccurrence_patterns(self, data_loader):
        """
        提取对象共现模式
        Args:
            data_loader: 数据加载器
        """
        print("提取共现模式...")
        
        for batch in tqdm(data_loader, desc="分析共现"):
            images = batch['images']

            # 提取对象及特征
            objects_list, _ = self.segmentation(images)
            relations_list = self.relation_extractor(objects_list)

            for objects, relations in zip(objects_list, relations_list):
                # 更新共现统计
                object_labels = [obj['label'] for obj in objects if obj['label'] in self.vocab_to_idx]

                for i, obj1 in enumerate(object_labels):
                    for obj2 in object_labels[i+1:]:
                        self.global_cooccurrence[obj1][obj2] += 1
                        self.global_cooccurrence[obj2][obj1] += 1

                # 更新因果统计（使用方向性关系）
                for subj_idx, _, obj_idx, conf in relations:
                    if subj_idx >= len(objects) or obj_idx >= len(objects):
                        continue

                    label_subj = objects[subj_idx]['label']
                    label_obj = objects[obj_idx]['label']

                    if label_subj not in self.vocab_to_idx or label_obj not in self.vocab_to_idx:
                        continue

                    self.causality_pairs[label_subj][label_obj] += float(conf)
    
    def train_gnn(
        self,
        data_loader,
        num_epochs=Config.NUM_EPOCHS,
        resume_path=None,
        disable_dropout: bool = False,
        target_mode: str = "binary",
    ):
        """
        训练GNN模型
        Args:
            data_loader: 数据加载器
            num_epochs: 训练轮数
        """
        print("开始训练GNN...")
        
        # === 错误原代码 ===
        # optimizer = optim.AdamW(
        #     self.gnn.parameters(),
        #     lr=Config.GNN_LEARNING_RATE,
        #     weight_decay=Config.GNN_WEIGHT_DECAY
        # )

        # === 修正后的代码 ===
        # 我们需要同时训练 GNN 和 分割模块中的特征构建器(Fusion MLP)
        # 注意：CLIP 模型本身在 core.txt 里已经冻结了 (requires_grad=False)，所以这里加入 feature_builder 只会训练 MLP
        params_to_optimize = list(self.gnn.parameters()) + \
                             list(self.segmentation.feature_builder.parameters())
        
        optimizer = optim.AdamW(
            params_to_optimize,
            lr=Config.GNN_LEARNING_RATE,
            weight_decay=Config.GNN_WEIGHT_DECAY
        )
        
        # 模式控制：overfit 时关闭 Dropout，避免输入/预测抖动导致难以“死记硬背”
        # 注意：eval() 不会阻止梯度回传到参数（只影响 Dropout/BN 行为）
        if disable_dropout:
            self.segmentation.feature_builder.eval()
            self.gnn.eval()
        else:
            self.segmentation.feature_builder.train()
            self.gnn.train()

        # ... 后续训练循环 ...

        start_epoch = 0
        # 断点恢复
        if resume_path is not None and os.path.isfile(resume_path):
            print(f"[断点恢复] 加载权重: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            if 'gnn_state_dict' in checkpoint:
                self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"✓ 已恢复到 epoch {start_epoch}")
            else:
                # 兼容只保存了 state_dict 的情况
                self.gnn.load_state_dict(checkpoint)
                print("✓ 仅恢复了模型参数 (无优化器/epoch 信息)")

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                images = batch['images']
                


                # # 提取对象和关系
                # objects_list = self.segmentation(images)
                # relations_list = self.relation_extractor(objects_list)
                
                # batch_loss = 0
                # valid_samples = 0
                
                # for objects, relations in zip(objects_list, relations_list):
                #     if len(objects) < 2:
                #         continue
                    
                #     # 构建图数据
                #     n_objects = len(objects)
                #     node_features = torch.randn(n_objects, Config.GNN_HIDDEN_DIM).to(self.device)




                # 提取对象和关系
                # 1. 【修改点】解包返回值：获取 features_list (CLIP特征)
                objects_list, features_list = self.segmentation(images)
                
                # 2. 【修改点】只把 objects_list 传给关系提取器
                relations_list = self.relation_extractor(objects_list)
                
                batch_loss = 0
                valid_samples = 0
                
                # 3. 【修改点】同时遍历 objects, relations 和 features
                for objects, relations, features in zip(objects_list, relations_list, features_list):
                    if len(objects) < 2:
                        continue
                    
                    # 构建图数据
                    n_objects = len(objects)
                    
                    # 4. 【修改点】使用 CLIP 提取的真实特征，而不是随机噪声(torch.randn)
                    # 之前的代码: node_features = torch.randn(...).to(self.device)
                    node_features = features  # 直接使用 model_core 返回的 Tensor
                    
                    # 如果 features 为空 (防止空图报错)，给一个零向量保护
                    if node_features.size(0) == 0:
                        node_features = torch.zeros(n_objects, Config.GNN_HIDDEN_DIM).to(self.device)
                    

                    
                    # 构建边
                    edges = []
                    edge_feats = []
                    for subj, rel, obj, conf in relations:
                        edges.append([subj, obj])
                        edge_feats.append([conf])
                    
                    if not edges:
                        continue
                    
                    edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
                    edge_features = torch.tensor(edge_feats, dtype=torch.float).to(self.device)
                    
                    # 前向传播
                    _, cooccur, causal = self.gnn(node_features, edge_index, edge_features)
                    
                    # 构建目标矩阵（使用当前图中的关系作为正样本，其他为负样本）
                    target_cooccur = torch.zeros(n_objects, n_objects).to(self.device)
                    target_causal = torch.zeros(n_objects, n_objects).to(self.device)

                    # 当前图的有向关系集合
                    relation_set = {(int(subj), int(obj)) for subj, _, obj, _ in relations}

                    if target_mode == "binary":
                        denom = max(n_objects * (n_objects - 1), 1)
                        density = len(relation_set) / denom
                        # 密度过高时，binary 目标几乎全为1，会导致 loss 虚低但泛化/推理无意义
                        if density > 0.5:
                            print(
                                f"⚠ 警告: 关系标签过稠密 (density={density:.2f}, edges={len(relation_set)}, N={n_objects})，"
                                "binary 训练可能出现伪收敛；建议改进关系抽取或使用 --target_mode soft。"
                            )

                    for i, obj_i in enumerate(objects):
                        for j, obj_j in enumerate(objects):
                            if i == j:
                                continue

                            if target_mode == "binary":
                                # 只有在本图中被抽取到的关系才记为正，其余视作负样本
                                target_cooccur[i, j] = 1.0 if (i, j) in relation_set else 0.0
                                target_causal[i, j] = 1.0 if (i, j) in relation_set else 0.0
                            else:
                                # soft 模式仍可使用全局统计
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

                    
                    # 计算损失
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
                print(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_epoch_loss:.4f}")

            # 每3个epoch保存一次权重（包含优化器和epoch信息）
            # if (epoch + 1) % 3 == 0:
            #     ckpt_dir = Config.CHECKPOINT_DIR
            #     os.makedirs(ckpt_dir, exist_ok=True)
            #     ckpt_path = os.path.join(ckpt_dir, f'gnn_epoch_{epoch+1}.pt')
            #     torch.save({
            #         'gnn_state_dict': self.gnn.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'epoch': epoch
            #     }, ckpt_path)
            
                # 每3个epoch保存一次权重
            if (epoch + 1) % 10 == 0:
                ckpt_dir = Config.CHECKPOINT_DIR
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f'new_gnn_epoch_{epoch+1}.pt')
                
                torch.save({
                    'gnn_state_dict': self.gnn.state_dict(),
                    # 新增：保存特征构建器的权重
                    'feature_builder_state_dict': self.segmentation.feature_builder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                }, ckpt_path)
                print(f"✓ 已保存GNN权重: {ckpt_path}")
    
    def aggregate_knowledge(self, data_loader):
        """
        聚合知识到矩阵形式
        Args:
            data_loader: 数据加载器
        """
        print("聚合知识库...")

        vocab_size = len(self.object_vocab)
        device = self.device

        self.gnn.eval()
        self.segmentation.feature_builder.eval()

        self.cooccurrence_matrix = torch.zeros(vocab_size, vocab_size, device=device)
        self.causality_matrix = torch.zeros(vocab_size, vocab_size, device=device)
        pair_counts = torch.zeros(vocab_size, vocab_size, device=device)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="聚合推断"):
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
                            # causal/cooccur 都是“当前图内”的局部矩阵，必须用 local_j 索引；
                            # 误用 global_j 会导致 IndexError（global 词表远大于当前图物体数）
                            self.causality_matrix[global_i, global_j] += causal[local_i, local_j]
                            pair_counts[global_i, global_j] += 1

                    if unknown_label_count > 0:
                        print(f"⚠ 聚合阶段遇到 {unknown_label_count} 个未知标签（不在词表中），已跳过")

        mask = pair_counts > 0
        if mask.any():
            self.cooccurrence_matrix[mask] /= pair_counts[mask]
            self.causality_matrix[mask] /= pair_counts[mask]

        self.cooccurrence_matrix = self.cooccurrence_matrix.cpu()
        self.causality_matrix = self.causality_matrix.cpu()

        print("✓ 知识库聚合完成")
    
    def save_knowledge_base(self, save_path):
        """
        保存知识库
        Args:
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 保存词汇表
        vocab_data = {
            'vocab': self.object_vocab,
            'vocab_to_idx': self.vocab_to_idx
        }
        with open(os.path.join(save_path, 'object_vocab.json'), 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        # 保存统计信息
        stats = {
            'global_cooccurrence': {k: dict(v) for k, v in self.global_cooccurrence.items()},
            'causality_pairs': {k: dict(v) for k, v in self.causality_pairs.items()}
        }
        with open(os.path.join(save_path, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 保存矩阵
        if self.cooccurrence_matrix is not None:
            np.save(
                os.path.join(save_path, 'cooccurrence_matrix.npy'),
                self.cooccurrence_matrix.cpu().numpy()
            )
        
        if self.causality_matrix is not None:
            np.save(
                os.path.join(save_path, 'causality_matrix.npy'),
                self.causality_matrix.cpu().numpy()
            )
        
        print(f"✓ 知识库已保存到: {save_path}")
    
    def load_knowledge_base(self, kb_path):
        """
        加载知识库
        Args:
            kb_path: 知识库目录路径
        """
        print(f"从 {kb_path} 加载知识库...")
        
        # 加载词汇表
        vocab_path = os.path.join(kb_path, 'object_vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                self.object_vocab = vocab_data['vocab']
                self.vocab_to_idx = vocab_data['vocab_to_idx']
            print(f"✓ 加载词汇表: {len(self.object_vocab)} 个对象")
        else:
            print(f"⚠ 词汇表文件不存在: {vocab_path}")
        
        # 加载统计信息
        stats_path = os.path.join(kb_path, 'statistics.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                # 重建defaultdict结构
                self.global_cooccurrence = defaultdict(lambda: defaultdict(float))
                self.causality_pairs = defaultdict(lambda: defaultdict(float))
                
                for obj1, obj2_dict in stats.get('global_cooccurrence', {}).items():
                    for obj2, count in obj2_dict.items():
                        self.global_cooccurrence[obj1][obj2] = count
                
                for obj1, obj2_dict in stats.get('causality_pairs', {}).items():
                    for obj2, count in obj2_dict.items():
                        self.causality_pairs[obj1][obj2] = count
            print(f"✓ 加载统计信息")
        else:
            print(f"⚠ 统计信息文件不存在: {stats_path}")
        
        # 加载共现矩阵
        cooccur_path = os.path.join(kb_path, 'cooccurrence_matrix.npy')
        if os.path.exists(cooccur_path):
            self.cooccurrence_matrix = torch.from_numpy(
                np.load(cooccur_path)
            ).to(Config.DEVICE)
            print(f"✓ 加载共现矩阵: {self.cooccurrence_matrix.shape}")
        else:
            print(f"⚠ 共现矩阵文件不存在: {cooccur_path}")
        
        # 加载因果矩阵
        causal_path = os.path.join(kb_path, 'causality_matrix.npy')
        if os.path.exists(causal_path):
            self.causality_matrix = torch.from_numpy(
                np.load(causal_path)
            ).to(Config.DEVICE)
            print(f"✓ 加载因果矩阵: {self.causality_matrix.shape}")
        else:
            print(f"⚠ 因果矩阵文件不存在: {causal_path}")
        
        print("✓ 知识库加载完成")
    
    def build(
        self,
        data_loader,
        num_epochs=Config.NUM_EPOCHS,
        resume_path=None,
        disable_dropout: bool = False,
        target_mode: str = "binary",
    ):
        """
        完整的知识库构建流程
        Args:
            data_loader: 数据加载器
            num_epochs: GNN训练轮数
        """
        # 1. 构建词汇表
        self.build_vocabulary(data_loader)
        
        # 2. 提取共现模式
        self.extract_cooccurrence_patterns(data_loader)
        
        # 3. 训练GNN
        self.train_gnn(
            data_loader,
            num_epochs,
            resume_path=resume_path,
            disable_dropout=disable_dropout,
            target_mode=target_mode,
        )
        
        # 4. 聚合知识
        self.aggregate_knowledge(data_loader)
        
        print("✓ 知识库构建完成!")


if __name__ == "__main__":
    # 测试知识库构建
    from data_loader import COCOCaptionDataset
    from torch.utils.data import DataLoader
    
    Config.create_dirs()
    
    # 创建数据集
    train_dataset = COCOCaptionDataset(
        split='train',
        max_samples=100
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: {
            'image': [item['image'] for item in x],
            'captions': [item['captions'] for item in x]
        }
    )
    
    # 构建知识库
    builder = KnowledgeBaseBuilder()
    builder.build(train_loader, num_epochs=3)
    
    # 保存知识库
    builder.save_knowledge_base(Config.KNOWLEDGE_BASE_DIR)
    
    print("\n测试知识库加载...")
    builder2 = KnowledgeBaseBuilder()
    builder2.load_knowledge_base(Config.KNOWLEDGE_BASE_DIR)
    
    print("✓ 所有测试通过!")
