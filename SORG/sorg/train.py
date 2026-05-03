import torch
import os
import argparse
from config import Config
from data_preparation import prepare_data
from knowledge_builder import KnowledgeBaseBuilder

def train_knowledge_base(args):
    """训练知识库和GNN模型"""
    
    print("="*60)
    print("视觉推理增强的多模态描述生成系统 - 训练模块")
    print("="*60)
    
    # 创建必要目录
    Config.create_dirs()
    Config.print_config()
    
    # 准备数据
    print("\n正在准备数据集...")
    train_loader, val_loader = prepare_data(
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 构建知识库
    print("\n开始构建知识库...")
    builder = KnowledgeBaseBuilder()
    builder.build(
        train_loader,
        num_epochs=args.num_epochs,
        resume_path=getattr(args, 'resume', None),
        target_mode=args.target_mode,
    )
    
    # 保存知识库与模型权重
    try:
        builder.save_knowledge_base(Config.KNOWLEDGE_DIR)
    except Exception as e:
        print(f"保存知识库失败: {e}")

    try:
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        model_path = os.path.join(Config.CHECKPOINT_DIR, 'gnn_state_5000.pt')

        # 保存 GNN 和特征融合 MLP，避免推理时随机初始化导致输出塌缩
        torch.save(
            {
                'gnn_state_dict': builder.gnn.state_dict(),
                'feature_builder_state_dict': builder.segmentation.feature_builder.state_dict(),
            },
            model_path,
        )
        print(f"已保存 GNN 模型权重到: {model_path}")
    except Exception as e:
        print(f"保存模型权重失败: {e}")

    print("\n训练完成!")
    print(f"模型保存在: {Config.CHECKPOINT_DIR}")
    print(f"知识库保存在: {Config.KNOWLEDGE_DIR}")


def main():
    parser = argparse.ArgumentParser(description='训练视觉推理增强模型')
    parser.add_argument('--max_train_samples', type=int, default=10000, help='最大训练样本数')
    parser.add_argument('--max_val_samples', type=int, default=1000, help='最大验证样本数')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--resume', type=str, default=None, help='断点恢复权重文件路径(.pt/.pth)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='模型权重保存目录')
    parser.add_argument('--target_mode', type=str, default='binary', choices=['binary', 'soft'], help='GNN训练目标模式')

    args = parser.parse_args()

    # 更新配置
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
    if args.checkpoint_dir:
        Config.CHECKPOINT_DIR = args.checkpoint_dir

    # 开始训练
    train_knowledge_base(args)


if __name__ == "__main__":
    main()
