import os
from pathlib import Path
from typing import Any, Dict
import torch
import yaml

class Config:
    PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    CONFIG_DIR = os.path.join(PROJECT_ROOT, 'configs')
    _loaded_files = []

    @classmethod
    def _read_yaml(cls, path: str) -> Dict[str, Any]:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = Path(cls.PROJECT_ROOT) / path_obj
        if not path_obj.exists():
            raise FileNotFoundError(f'Config file not found: {path_obj}')
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        cls._loaded_files.append(str(path_obj))
        return data

    @classmethod
    def _flatten(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict):
                flat.update(cls._flatten(value))
            else:
                flat[key.upper()] = value
        return flat

    @classmethod
    def _resolve_path(cls, value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        if value == '':
            return value
        if os.path.isabs(value):
            return value
        if '://' in value:
            return value
        return os.path.normpath(os.path.join(cls.PROJECT_ROOT, value))

    @classmethod
    def _path_keys(cls):
        return {'DATA_DIR', 'COCO_DIR', 'NLTK_DATA_DIR', 'COCO_ROOT', 'COCO_TRAIN_IMG', 'COCO_VAL_IMG', 'COCO_IMAGES', 'COCO_VAL_IMAGES', 'COCO_TRAIN_ANN', 'COCO_VAL_ANN', 'COCO_ANNOTATIONS', 'MODEL_DIR', 'CHECKPOINT_DIR', 'KNOWLEDGE_BASE_DIR', 'KNOWLEDGE_DIR', 'RESULTS_DIR', 'LLAVA_MODEL_PATH', 'MINIGPT4_MODEL_PATH', 'INSTRUCTBLIP_MODEL_PATH', 'QWEN_MODEL_PATH', 'LLM_MODEL_NAME'}

    @classmethod
    def apply(cls, values: Dict[str, Any]) -> None:
        for key, value in cls._flatten(values).items():
            if key == 'PROJECT_ROOT':
                continue
            if key == 'DEVICE' and value == 'auto':
                value = 'cuda' if torch.cuda.is_available() else 'cpu'
            if key in cls._path_keys():
                value = cls._resolve_path(value)
            setattr(cls, key, value)
        cls.KNOWLEDGE_DIR = getattr(cls, 'KNOWLEDGE_DIR', getattr(cls, 'KNOWLEDGE_BASE_DIR', None))
        cls.COCO_IMAGES = getattr(cls, 'COCO_IMAGES', getattr(cls, 'COCO_TRAIN_IMG', None))
        cls.COCO_VAL_IMAGES = getattr(cls, 'COCO_VAL_IMAGES', getattr(cls, 'COCO_VAL_IMG', None))
        cls.LLM_MODEL_NAME = getattr(cls, 'LLM_MODEL_NAME', getattr(cls, 'LLAVA_MODEL_PATH', None))

    @classmethod
    def load_config(cls, config_file: str=None) -> None:
        cls._loaded_files = []
        cls.apply(cls._read_yaml(os.path.join('configs', 'default.yaml')))
        env_config = os.environ.get('VCC_CONFIG')
        if env_config:
            cls.apply(cls._read_yaml(env_config))
        if config_file:
            cls.apply(cls._read_yaml(config_file))

    @staticmethod
    def create_dirs() -> None:
        for path in [Config.CHECKPOINT_DIR, Config.KNOWLEDGE_BASE_DIR, Config.RESULTS_DIR, Config.MODEL_DIR, Config.NLTK_DATA_DIR]:
            os.makedirs(path, exist_ok=True)

    @classmethod
    def print_config(cls) -> None:
        print('=' * 60)
        print('Configuration')
        print('=' * 60)
        print(f"Loaded config files: {', '.join(cls._loaded_files)}")
        print(f'Device: {cls.DEVICE}')
        print(f'Project root: {cls.PROJECT_ROOT}')
        print('\nData paths:')
        print(f'  - COCO train images: {cls.COCO_TRAIN_IMG}')
        print(f'  - COCO validation images: {cls.COCO_VAL_IMG}')
        print(f'  - COCO annotations: {cls.COCO_ANNOTATIONS}')
        print('\nModel paths:')
        print(f'  - Segmentation model: {cls.SEGMENTATION_MODEL}')
        print(f'  - Model type: {cls.MODEL_TYPE}')
        print(f'  - LLaVA model: {cls.LLAVA_MODEL_PATH}')
        print('\nOutput paths:')
        print(f'  - Checkpoints: {cls.CHECKPOINT_DIR}')
        print(f'  - Knowledge base: {cls.KNOWLEDGE_BASE_DIR}')
        print(f'  - Results: {cls.RESULTS_DIR}')
        print('\nTraining configuration:')
        print(f'  - Batch size: {cls.BATCH_SIZE}')
        print(f'  - Epochs: {cls.NUM_EPOCHS}')
        print(f'  - Learning rate: {cls.LEARNING_RATE}')
        print('\nGNN configuration:')
        print(f'  - Hidden dimension: {cls.GNN_HIDDEN_DIM}')
        print(f'  - Layers: {cls.GNN_NUM_LAYERS}')
        print(f'  - Attention heads: {cls.GNN_NUM_HEADS}')
        print(f'  - Dropout: {cls.GNN_DROPOUT}')
        print('=' * 60)

    @classmethod
    def verify_paths(cls) -> bool:
        print('\nVerifying paths...')
        paths_to_check = {'COCO train images': cls.COCO_TRAIN_IMG, 'COCO validation images': cls.COCO_VAL_IMG, 'COCO annotations': cls.COCO_ANNOTATIONS}
        all_exist = True
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            status = 'OK' if exists else 'MISSING'
            print(f'{status}: {name}: {path}')
            if not exists:
                all_exist = False
        if os.path.exists(cls.LLM_MODEL_NAME):
            print(f'OK: Local VLM model: {cls.LLM_MODEL_NAME}')
        else:
            print(f'Info: local VLM model path not found: {cls.LLM_MODEL_NAME}')
            print('The model loader may download from Hugging Face if this value is a repository ID.')
        print('\nAll required paths are available.' if all_exist else '\nSome required paths are missing.')
        return all_exist
Config.load_config()
if __name__ == '__main__':
    Config.create_dirs()
    Config.print_config()
    Config.verify_paths()
