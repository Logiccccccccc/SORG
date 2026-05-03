"""全局配置文件"""
import os
import torch


class Config:
    """全局配置类"""
    
    # ============ 设备配置 ============
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ============ 路径配置 ============
    PROJECT_ROOT = '/root/autodl-tmp/HuL_Code/vision-caption-correction' # 请根据实际路径修改，此处为服务器路径
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    COCO_DIR = os.path.join(DATA_DIR, 'coco')

    # NLTK 数据目录（建议放在项目 data 下，避免因环境/容器变化导致 ~/.nltk_data 丢失）
    NLTK_DATA_DIR = os.path.join(DATA_DIR, 'nltk_data')
    
    # COCO数据集路径
    COCO_ROOT = COCO_DIR
    COCO_TRAIN_IMG = os.path.join(COCO_DIR, 'train2017')
    COCO_VAL_IMG = os.path.join(COCO_DIR, 'val2017')
    COCO_IMAGES = COCO_TRAIN_IMG  # 向后兼容
    COCO_VAL_IMAGES = COCO_VAL_IMG  # 向后兼容
    COCO_TRAIN_ANN = os.path.join(COCO_DIR, 'annotations', 'captions_train2017.json')
    COCO_VAL_ANN = os.path.join(COCO_DIR, 'annotations', 'captions_val2017.json')
    COCO_ANNOTATIONS = os.path.join(COCO_DIR, 'annotations')  # 向后兼容
    
    # 模型和输出路径
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, 'knowledge_base')
    KNOWLEDGE_DIR = KNOWLEDGE_BASE_DIR  # 向后兼容
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    
    # ============ 模型配置 ============
    # 分割模型
    SEGMENTATION_MODEL = "facebook/maskformer-swin-large-coco"
    
    # LLaVA模型配置
    MODEL_TYPE = "llava" # 可选: "llava", "minigpt4", "instructblip", "qwen"
    LLAVA_MODEL_PATH = "/root/autodl-tmp/HuL_Code/vision-caption-correction/models/llava-1.5-7b-hf"
    
    # MiniGPT-4 / BLIP-2 Path
    MINIGPT4_MODEL_PATH = "/root/autodl-tmp/HuL_Code/vision-caption-correction/models/blip2-opt-2.7b" 
    
    # InstructBLIP Path
    INSTRUCTBLIP_MODEL_PATH = "/root/autodl-tmp/HuL_Code/vision-caption-correction/models/instructblip-vicuna-7b"

    # Qwen-VL Path (ModelScope)
    QWEN_MODEL_PATH = "/root/autodl-tmp/HuL_Code/vision-caption-correction/models/qwen3-vl-8b-instruct-abliterated-CV"

    LLM_MODEL_NAME = os.path.join(MODEL_DIR, 'llava-1.5-7b-hf')  # 本地路径（如果需要）
    LLM_CACHE_DIR = None  # 使用本地模型时不需要缓存目录
    
    # ============ GNN配置 ============
    GNN_HIDDEN_DIM = 512
    GNN_NUM_LAYERS = 3
    GNN_NUM_HEADS = 8
    GNN_DROPOUT = 0.1
    GNN_LEARNING_RATE = 1e-4
    GNN_WEIGHT_DECAY = 1e-5
    
    # ============ 训练配置 ============
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    
    # ============ 核心对象选择配置 ============
    MAX_OBJECTS = 30
    MIN_CORE_OBJECTS = 3
    MAX_CORE_OBJECTS = 10 #最多选几个核心对象

    # ============ 关系抽取(训练/推理一致) ============
    REL_MAX_PER_OBJECT = 3  #每个节点最多连多少个边，这是为了控制图的稀疏度，要不然每个节点都互相连接，那生成的共现矩阵（binary方法）就会成全1了，GNN训练的时候就会认为所有物体都共现，后面生成的inferred就会非常多，诱发幻觉
    REL_MIN_PAIR_SCORE = 0.40
    REL_NEAR_THRESHOLD = 0.20
    
    # ============ 修正阈值配置 ============
    CONFIDENCE_LOW_THRESHOLD = 0.4
    CONFIDENCE_HIGH_THRESHOLD = 0.7
    
    # ============ 不确定性估计配置 ============
    MC_DROPOUT_SAMPLES = 10
    INFERENCE_CONFIDENCE_THRESHOLD = 0.5
    MAX_INFERRED_OBJECTS = 8 # 每张图最多推理补全8个对象，为了过滤背景，背景对推理作用不大，反而会激发llava的语言先验，增加幻觉

    # ============ 视觉上下文(给 LLaVA)过滤策略 ============
    # 背景类/Stuff 类如果进入 visual_context，容易诱发“场景先验补全”。
    # 但过滤太狠会让可用对象名词过少，导致描述质量下降。
    # 下面开关用于做对照实验：
    # - 保守：两者都 True（默认，行为与当前一致）
    # - 更宽松：仅过滤 merged/stuff/other（KEYWORDS=False）
    # - 最宽松：两者都 False（不建议直接默认）


    #两个True说明过滤程度比较高，两个都False说明不过滤
    VISUAL_CONTEXT_FILTER_STUFF_MERGED = True    #为 True 时，把包含 merged/stuff/other 的标签当背景
    VISUAL_CONTEXT_FILTER_KEYWORDS = True
    VISUAL_CONTEXT_BACKGROUND_KEYWORDS = (
        'wall', 'floor', 'ceiling', 'pavement', 'road', 'grass', 'sky', 'building',
        'mountain', 'dirt', 'curtain', 'rug', 'banner'
    )

    # 传给 LLaVA 的关系数量：只取置信度最高的 Top-K 条关系（只围绕核心对象）
    # 用于对照实验：K 越大上下文越丰富，但也更可能引入噪声关系
    MAX_RELATIONS_IN_CONTEXT = 5
    
    # ============ 评估配置 ============
    EVAL_METRICS = ['bleu', 'rouge', 'meteor', 'chair']
    
    # ============ 数据加载配置 ============
    NUM_WORKERS = 4
    
    @staticmethod
    def create_dirs():
        """创建必要的目录"""
        import os
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.KNOWLEDGE_BASE_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.NLTK_DATA_DIR, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("配置信息")
        print("=" * 60)
        print(f"设备: {cls.DEVICE}")
        print(f"项目根目录: {cls.PROJECT_ROOT}")
        print(f"\n数据路径:")
        print(f"  - COCO训练图像: {cls.COCO_TRAIN_IMG}")
        print(f"  - COCO验证图像: {cls.COCO_VAL_IMG}")
        print(f"  - COCO标注: {cls.COCO_ANNOTATIONS}")
        print(f"\n模型路径:")
        print(f"  - 分割模型: {cls.SEGMENTATION_MODEL}")
        print(f"  - LLaVA模型: {cls.LLAVA_MODEL_PATH}")
        print(f"\n输出路径:")
        print(f"  - 检查点: {cls.CHECKPOINT_DIR}")
        print(f"  - 知识库: {cls.KNOWLEDGE_BASE_DIR}")
        print(f"  - 结果: {cls.RESULTS_DIR}")
        print(f"\n训练配置:")
        print(f"  - 批次大小: {cls.BATCH_SIZE}")
        print(f"  - 训练轮数: {cls.NUM_EPOCHS}")
        print(f"  - 学习率: {cls.LEARNING_RATE}")
        print(f"\nGNN配置:")
        print(f"  - 隐藏维度: {cls.GNN_HIDDEN_DIM}")
        print(f"  - 层数: {cls.GNN_NUM_LAYERS}")
        print(f"  - 注意力头数: {cls.GNN_NUM_HEADS}")
        print(f"  - Dropout: {cls.GNN_DROPOUT}")
        print("=" * 60)
    
    @classmethod
    def verify_paths(cls):
        """验证路径是否存在"""
        print("\n验证路径...")
        paths_to_check = {
            'COCO训练图像': cls.COCO_TRAIN_IMG,
            'COCO验证图像': cls.COCO_VAL_IMG,
            'COCO标注': cls.COCO_ANNOTATIONS,
        }
        
        all_exist = True
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            status = "✓" if exists else "✗"
            print(f"{status} {name}: {path}")
            if not exists:
                all_exist = False
                print(f"   警告: 路径不存在!")
        
        # 检查LLaVA模型（可能是远程或本地）
        if os.path.exists(cls.LLM_MODEL_NAME):
            print(f"✓ LLaVA模型(本地): {cls.LLM_MODEL_NAME}")
        else:
            print(f"ℹ LLaVA模型(远程): {cls.LLAVA_MODEL_PATH}")
            print(f"   将从Hugging Face下载")
        
        if all_exist:
            print("\n✓ 所有必要路径验证通过!")
        else:
            print("\n✗ 部分路径不存在，请检查!")
        
        return all_exist


if __name__ == "__main__":
    # 测试配置
    Config.create_dirs()
    Config.print_config()
    Config.verify_paths()
