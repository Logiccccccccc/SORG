import os
import json
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import Config

class COCODataset(Dataset):
    """COCO数据集加载器"""
    
    def __init__(self, root_dir, ann_file, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        
        print(f"加载COCO标注文件: {ann_file}")
        self.coco = COCO(ann_file)
        
        # 只使用有caption的图像
        self.img_ids = list(self.coco.imgs.keys())
        if max_samples:
            self.img_ids = self.img_ids[:max_samples]
        
        print(f"加载了 {len(self.img_ids)} 张图像")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # 加载图像
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像不存在 {img_path}")
            # 返回一个空白图像
            image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.open(img_path).convert('RGB')
        
        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 提取对象类别
        categories = []
        bboxes = []
        for ann in anns:
            if 'category_id' in ann:
                cat_name = self.coco.loadCats(ann['category_id'])[0]['name']
                categories.append(cat_name)
                if 'bbox' in ann:
                    bboxes.append(ann['bbox'])
        
        # 加载captions
        cap_ids = self.coco.getAnnIds(imgIds=img_id)
        caps = self.coco.loadAnns(cap_ids)
        captions = [cap['caption'] for cap in caps if 'caption' in cap]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image_id': img_id,
            'image': image,
            'categories': categories,
            'bboxes': bboxes,
            'captions': captions,
            'file_name': img_info['file_name']
        }


def collate_fn(batch):
    """自定义批处理函数"""
    return {
        'image_ids': [item['image_id'] for item in batch],
        'images': [item['image'] for item in batch],
        'categories': [item['categories'] for item in batch],
        'bboxes': [item['bboxes'] for item in batch],
        'captions': [item['captions'] for item in batch],
        'file_names': [item['file_name'] for item in batch]
    }


def prepare_data(max_train_samples=10000, max_val_samples=1000):
    """准备数据加载器"""
    
    print("\n正在准备数据集...")
    
    # 使用已存在的标注文件
    train_ann_file = os.path.join(Config.COCO_ANNOTATIONS, 
                                   'instances_train2017.json')
    val_ann_file = os.path.join(Config.COCO_ANNOTATIONS, 
                                 'instances_val2017.json')
    
    # 验证文件存在
    if not os.path.exists(train_ann_file):
        raise FileNotFoundError(f"训练标注文件不存在: {train_ann_file}")
    if not os.path.exists(val_ann_file):
        raise FileNotFoundError(f"验证标注文件不存在: {val_ann_file}")
    
    print(f"训练图像目录: {Config.COCO_IMAGES}")
    print(f"验证图像目录: {Config.COCO_VAL_IMAGES}")
    
    train_dataset = COCODataset(
        root_dir=Config.COCO_IMAGES,
        ann_file=train_ann_file,
        max_samples=max_train_samples
    )
    
    val_dataset = COCODataset(
        root_dir=Config.COCO_VAL_IMAGES,
        ann_file=val_ann_file,
        max_samples=max_val_samples
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    Config.create_dirs()
    Config.verify_paths()
    
    train_loader, val_loader = prepare_data(max_train_samples=100, 
                                             max_val_samples=20)
    print(f"\n训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 测试加载一个批次
    batch = next(iter(train_loader))
    print(f"\n批次大小: {len(batch['images'])}")
    print(f"第一张图像的类别: {batch['categories'][0][:5]}")
