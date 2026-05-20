import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
import json
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import Config

class COCODataset(Dataset):

    def __init__(self, root_dir, ann_file, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        print(f'Loading COCO annotation file: {ann_file}')
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        if max_samples:
            self.img_ids = self.img_ids[:max_samples]
        print(f'Loaded {len(self.img_ids)} images')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f'Warning: image does not exist {img_path}')
            image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.open(img_path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        categories = []
        bboxes = []
        for ann in anns:
            if 'category_id' in ann:
                cat_name = self.coco.loadCats(ann['category_id'])[0]['name']
                categories.append(cat_name)
                if 'bbox' in ann:
                    bboxes.append(ann['bbox'])
        cap_ids = self.coco.getAnnIds(imgIds=img_id)
        caps = self.coco.loadAnns(cap_ids)
        captions = [cap['caption'] for cap in caps if 'caption' in cap]
        if self.transform:
            image = self.transform(image)
        return {'image_id': img_id, 'image': image, 'categories': categories, 'bboxes': bboxes, 'captions': captions, 'file_name': img_info['file_name']}

def collate_fn(batch):
    return {'image_ids': [item['image_id'] for item in batch], 'images': [item['image'] for item in batch], 'categories': [item['categories'] for item in batch], 'bboxes': [item['bboxes'] for item in batch], 'captions': [item['captions'] for item in batch], 'file_names': [item['file_name'] for item in batch]}

def prepare_data(max_train_samples=10000, max_val_samples=1000):
    print('\nPreparing dataset...')
    train_ann_file = os.path.join(Config.COCO_ANNOTATIONS, 'instances_train2017.json')
    val_ann_file = os.path.join(Config.COCO_ANNOTATIONS, 'instances_val2017.json')
    if not os.path.exists(train_ann_file):
        raise FileNotFoundError(f'Training annotation file does not exist: {train_ann_file}')
    if not os.path.exists(val_ann_file):
        raise FileNotFoundError(f'Validation annotation file does not exist: {val_ann_file}')
    print(f'Training image directory: {Config.COCO_IMAGES}')
    print(f'Validation image directory: {Config.COCO_VAL_IMAGES}')
    train_dataset = COCODataset(root_dir=Config.COCO_IMAGES, ann_file=train_ann_file, max_samples=max_train_samples)
    val_dataset = COCODataset(root_dir=Config.COCO_VAL_IMAGES, ann_file=val_ann_file, max_samples=max_val_samples)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, collate_fn=collate_fn)
    return (train_loader, val_loader)
if __name__ == '__main__':
    Config.create_dirs()
    Config.verify_paths()
    train_loader, val_loader = prepare_data(max_train_samples=100, max_val_samples=20)
    print(f'\nTraining batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')
    batch = next(iter(train_loader))
    print(f"\nBatch size: {len(batch['images'])}")
    print(f"Categories of the first image: {batch['categories'][0][:5]}")
