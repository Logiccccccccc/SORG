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
import argparse
from config import Config
from data_preparation import prepare_data
from knowledge_builder import KnowledgeBaseBuilder

def train_knowledge_base(args):
    print('=' * 60)
    print('Vision-reasoning-enhanced multimodal caption generation system - training module')
    print('=' * 60)
    Config.create_dirs()
    Config.print_config()
    print('\nPreparing dataset...')
    train_loader, val_loader = prepare_data(max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples)
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')
    print('\nBuilding knowledge base...')
    builder = KnowledgeBaseBuilder()
    builder.build(train_loader, num_epochs=args.num_epochs, resume_path=getattr(args, 'resume', None), target_mode=args.target_mode)
    try:
        builder.save_knowledge_base(Config.KNOWLEDGE_DIR)
    except Exception as e:
        print(f'Failed to save knowledge base: {e}')
    try:
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        model_path = os.path.join(Config.CHECKPOINT_DIR, 'gnn_state_5000.pt')
        torch.save({'gnn_state_dict': builder.gnn.state_dict(), 'feature_builder_state_dict': builder.segmentation.feature_builder.state_dict()}, model_path)
        print(f'Saved GNN model weights to: {model_path}')
    except Exception as e:
        print(f'Failed to save model weights: {e}')
    print('\nTraining completed.')
    print(f'Model saved in: {Config.CHECKPOINT_DIR}')
    print(f'Knowledge base saved in: {Config.KNOWLEDGE_DIR}')

def main():
    parser = argparse.ArgumentParser(description='Train the vision-reasoning-enhanced model.')
    parser.add_argument('--max_train_samples', type=int, default=10000, help='Maximum number of training samples.')
    parser.add_argument('--max_val_samples', type=int, default=1000, help='Maximum number of validation samples.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate.')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path for resuming training (.pt/.pth).')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for saving model checkpoints.')
    parser.add_argument('--target_mode', type=str, default='binary', choices=['binary', 'soft'], help='GNN training target mode.')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to the YAML training config')
    args = parser.parse_args()
    Config.load_config(args.config)
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
        Config.GNN_LEARNING_RATE = args.learning_rate
    if args.checkpoint_dir:
        Config.CHECKPOINT_DIR = args.checkpoint_dir
    train_knowledge_base(args)
if __name__ == '__main__':
    main()
