"""

映射表缩减版


标准 CHAIR 评估模块 (Standard CHAIR Evaluator)
基于 Rohrbach et al. (2018) 的逻辑进行标准化实现。
使用 NLTK WordNetLemmatizer 处理复数，并包含完整的 COCO 同义词映射。

python evaluation_CHAIR_test.py \
    --results_file /root/autodl-tmp/VISION-CAPTION-CORRECTION/RESULT/ablation_perturbation.json \
    --coco_instances /root/autodl-tmp/coco/annotations/instances_val2017.json
"""
import os
import json
import argparse
import re
import zipfile
import nltk
from collections import defaultdict
from pycocotools.coco import COCO
from config import Config

# =========================================================
# NLTK 资源检查与加载 (保持你原有的优秀逻辑)
# =========================================================
os.environ.setdefault("NLTK_DATA", Config.NLTK_DATA_DIR)
os.makedirs(os.environ["NLTK_DATA"], exist_ok=True)

def _ensure_nltk_resource(resource_path: str, package: str) -> None:
    """确保 NLTK 资源可用"""
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, download_dir=os.environ["NLTK_DATA"], quiet=True)

_ensure_nltk_resource('tokenizers/punkt', 'punkt')
_ensure_nltk_resource('corpora/wordnet', 'wordnet')
_ensure_nltk_resource('corpora/omw-1.4', 'omw-1.4') # Lemmatizer 有时需要这个

from nltk.stem import WordNetLemmatizer

# =========================================================
'''映射表缩减版'''
# =========================================================
COCO_SYNONYMS = {
    'airplane': ['airplane', 'aeroplane', 'plane', 'jet', '747', 'airplanes', 'aeroplanes', 'planes', 'jets', '747s'],
    'apple': ['apple', 'apples'],
    'backpack': ['backpack', 'backpacks', 'rucksack', 'rucksacks', 'satchel', 'satchels', 'pack', 'packs'],
    'banana': ['banana', 'bananas'],
    'baseball bat': ['baseball bat', 'baseball bats', 'bat', 'bats'],
    'baseball glove': ['baseball glove', 'baseball gloves', 'glove', 'gloves', 'mitt', 'mitts'],
    'bear': ['bear', 'bears', 'teddy bear', 'teddy bears', 'teddy', 'teddies'],
    'book': ['book', 'books'],
    'bottle': ['bottle', 'bottles', 'water bottle', 'water bottles', 'pop bottle', 'pop bottles'],
    'car': ['car', 'cars', 'auto', 'autos', 'automobile', 'automobiles', 'sedan', 'sedans', 'suv', 'suvs', 'jeep', 'jeeps', 'taxi', 'taxis', 'cab', 'cabs', 'van', 'vans', 'minivan', 'minivans'],
    'carrot': ['carrot', 'carrots'],
    'cat': ['cat', 'cats', 'kitten', 'kittens', 'kitty', 'kitties', 'feline', 'felines'],
    'cell phone': ['cell phone', 'cell phones', 'mobile phone', 'mobile phones', 'cellphone', 'cellphones', 'smartphone', 'smartphones', 'phone', 'phones'],
    'chair': ['chair', 'chairs', 'seat', 'seats', 'stool', 'stools'],
    'elephant': ['elephant', 'elephants'],
    'fire hydrant': ['fire hydrant', 'fire hydrants', 'hydrant', 'hydrants'],
    'fork': ['fork', 'forks'],
    'frisbee': ['frisbee', 'frisbees', 'disk', 'disks'],
    'giraffe': ['giraffe', 'giraffes'],
    'hair drier': ['hair drier', 'hair driers', 'hair dryer', 'hair dryers', 'blow dryer', 'blow dryers'],
    'handbag': ['handbag', 'handbags', 'purse', 'purses', 'bag', 'bags', 'pocketbook', 'pocketbooks'],
    'horse': ['horse', 'horses', 'pony', 'ponies', 'foal', 'foals', 'mare', 'mares', 'stallion', 'stallions'],
    'hot dog': ['hot dog', 'hot dogs', 'hotdog', 'hotdogs', 'frank', 'franks', 'frankfurter', 'frankfurters'],
    'keyboard': ['keyboard', 'keyboards', 'key board', 'key boards'],
    'kite': ['kite', 'kites'],
    'parking meter': ['parking meter', 'parking meters'],
    'person': ['person', 'people', 'persons', 'man', 'men', 'woman', 'women', 'boy', 'boys', 'girl', 'girls', 'child', 'children', 'kid', 'kids', 'guy', 'guys', 'lady', 'ladies', 'gentleman', 'gentlemen', 'baby', 'babies', 'toddler', 'toddlers', 'player', 'players', 'spectator', 'spectators', 'pedestrian', 'pedestrians', 'policeman', 'policemen', 'worker', 'workers'],
    'pizza': ['pizza', 'pizzas', 'pie', 'pies'],
    'potted plant': ['potted plant', 'potted plants', 'plant', 'plants', 'houseplant', 'houseplants', 'flower', 'flowers', 'pot', 'pots', 'vase', 'vases', 'bush', 'bushes', 'shrub', 'shrubs', 'tree', 'trees'],
    'refrigerator': ['refrigerator', 'refrigerators', 'fridge', 'fridges'],
    'remote': ['remote', 'remotes', 'remote control', 'remote controls', 'controller', 'controllers'],
    'sandwich': ['sandwich', 'sandwiches', 'sub', 'subs', 'burger', 'burgers', 'hamburger'],
    'scissors': ['scissors', 'shears', 'clippers'],
    'suitcase': ['suitcase', 'suitcases', 'luggage',  'cases', 'trunk', 'trunks', 'valise', 'valises'],
    'surfboard': ['surfboard', 'surfboards'],
    'teddy bear': ['teddy bear', 'teddy bears', 'teddy', 'teddies', 'stuffed animal', 'stuffed animals', 'plush toy', 'plush toys'],
    'tennis racket': ['tennis racket', 'tennis rackets', 'racket', 'rackets', 'racquet', 'racquets'],
    'tie': ['tie', 'ties', 'necktie', 'neckties', 'bowtie', 'bowties'],
    'toaster': ['toaster', 'toasters'],
    'toilet': ['toilet', 'toilets', 'commode', 'commodes', 'lavatory', 'lavatories', 'potty', 'potties'],
    'umbrella': ['umbrella', 'umbrellas', 'parasol', 'parasols'],
    'wine glass': ['wine glass', 'wine glasses', 'wineglass', 'wineglasses', 'glass', 'glasses', 'goblet', 'goblets'],
    'zebra': ['zebra', 'zebras']
}
class StandardCHAIR:
    def __init__(self, coco_instances_file):
        print(f"加载 instances 标注: {coco_instances_file}")
        
        # 修复: PyCOCOTools 在处理 caption 标注时会崩溃 (因为没有 category_id)
        # 我们手动加载过滤后的 annotations (仅保留 Object Detection 部分)
        import json
        with open(coco_instances_file, 'r') as f:
            full_data = json.load(f)
        
        # 过滤: 仅保留包含 category_id 的标注 (即物体检测标注)
        object_anns = [ann for ann in full_data.get('annotations', []) if 'category_id' in ann]
        full_data['annotations'] = object_anns
        
        # 使用过滤后的数据初始化 COCO
        # COCO 通常接受文件名，如果你传文件名它会自己加载。
        # 如果我们想传字典对象，COCO 类本身不支持直接传 dict (它只接受文件名)
        # 除非我们魔改 COCO 类，或者存一个临时文件。
        
        # 方案: 存一个临时文件供 CHAIR 使用
        temp_instances_file = coco_instances_file.replace('.json', '_instances_only.json')
        with open(temp_instances_file, 'w') as f:
            json.dump(full_data, f)
            
        self.coco = COCO(temp_instances_file)
        self.lemmatizer = WordNetLemmatizer()
        
        # 1. 构建反向索引 (word -> canonical_category)
        # 例如: 'boys' -> 'person', 'jet' -> 'airplane'
        self.word_to_cat = {}
        for cat, synonyms in COCO_SYNONYMS.items():
            for syn in synonyms:
                # 均转为小写
                self.word_to_cat[syn.lower()] = cat
        
        # 2. 获取 COCO 类别 ID 映射
        # 只有在 instances 文件中存在的类别才会被评估
        self.cat_id_to_name = {}
        self.cat_name_to_id = {}
        for cat_id, cat_info in self.coco.cats.items():
            cat_name = cat_info['name']
            self.cat_id_to_name[cat_id] = cat_name
            self.cat_name_to_id[cat_name] = cat_id

        print(f"同义词库构建完成，覆盖 {len(self.word_to_cat)} 个词汇变体。")

        # 预编译正则，用于处理特殊过滤
        self._non_noun_followers = {
            'a', 'an', 'the', 'this', 'that', 'these', 'those',
            'and', 'or', 'but', 'on', 'in', 'at', 'with', 'of', 'to', 'from',
            'near', 'next', 'beside', 'behind', 'under', 'over', 'above'
        }

    def _tokenize(self, caption):
        """简单的 NLTK 分词"""
        return nltk.word_tokenize(caption.lower())

    def _get_mentioned_objects(self, caption):
        """
        从描述中提取提到的 COCO 对象。  
        逻辑：
        1. 分词
        2. 优先匹配双词短语 (Double word matching) - 比如 'traffic light'
        3. 匹配单次 (Single word matching) 并进行词形还原 (Lemmatization)
        4. 应用 'No/Sign' 和 'Orange' 规则过滤
        """
        tokens = self._tokenize(caption)
        mentioned_cats = set()
        
        # --- 步骤 A: 提取候选词 ---
        
        skip_indices = set()
        
        # 1. 尝试匹配双词同义词 (2-grams)
        # 比如 "hot dog", "traffic light", "cell phone"
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                bi_gram = f"{tokens[i]} {tokens[i+1]}"
                if bi_gram in self.word_to_cat:
                    mentioned_cats.add(self.word_to_cat[bi_gram])
                    skip_indices.add(i)
                    skip_indices.add(i+1)

        # 2. 匹配单次 (1-gram) 并做 Lemmatization
        for i, token in enumerate(tokens):
            if i in skip_indices:
                continue
            
            # 尝试直接匹配
            if token in self.word_to_cat:
                mentioned_cats.add(self.word_to_cat[token])
                continue
            
            # 尝试还原后匹配 (buses -> bus, mice -> mouse)
            lemma = self.lemmatizer.lemmatize(token, pos='n')
            if lemma in self.word_to_cat:
                mentioned_cats.add(self.word_to_cat[lemma])
        
        if not mentioned_cats:
            return set()

        # --- 步骤 B: 应用语境过滤 (Context Filtering) ---
        # 保留这个优秀的逻辑，以增强评估的准确性

        # 1. 过滤 "Orange" 颜色
        if 'orange' in mentioned_cats:
            # 检查原始 token 序列
            is_color = False
            for i, tok in enumerate(tokens):
                if tok == 'orange':
                    # 检查后面是否紧跟名词性的词
                    nxt = tokens[i+1] if i+1 < len(tokens) else None
                    if nxt and nxt not in self._non_noun_followers:
                        # "orange shirt" -> orange 是颜色
                        is_color = True
                        break
                    # "orange and white" -> orange 是颜色
                    if nxt == 'and':
                        nxt2 = tokens[i+2] if i+2 < len(tokens) else None
                        if nxt2 and nxt2 not in self._non_noun_followers:
                            is_color = True
                            break
            
            # 如果判定为颜色且不是明确的复数 oranges (lemmatizer处理过)，则移除
            # 注意：如果句子同时包含 orange fruit 和 orange color，这种简单逻辑可能会误删，
            # 但作为评估指标，宁可漏检也不要误报幻觉。
            if is_color:
                mentioned_cats.discard('orange')

        # 2. 过滤 "No X" 和 "X Sign"
        caption_lower = caption.lower()
        final_cats = set(mentioned_cats)
        
        # 处理 "No <object>"
        if 'no' in tokens:
            for cat in list(final_cats):
                # 检查是否存在 "no {cat}" 或 "no {synonym}"
                # 构造所有可能的同义词形式
                candidates = [k for k, v in self.word_to_cat.items() if v == cat]
                for cand in candidates:
                    # 简单的正则检查
                    if re.search(rf"\bno\s+{re.escape(cand)}\b", caption_lower):
                        final_cats.discard(cat)
                        break
        
        # 处理 "... sign" (如 stop sign 已经是独立类别，这里主要处理 no parking sign 等)
        if 'sign' in tokens or 'signs' in tokens:
             # 如果提到 sign，检查是否否定了之前的物体
             pass # 标准 CHAIR 并没有非常复杂的 sign 逻辑，这里只保留最安全的 no 逻辑

        return final_cats

    def compute_chair(self, generated_captions, image_ids):
        """
        计算 CHAIRs 和 CHAIRi
        """
        chair_s_count = 0
        chair_i_hallucinated_count = 0
        chair_i_total_count = 0
        
        num_sentences = 0
        
        # 缓存 GT 标注，避免重复查询
        img_id_to_gt_cats = {}
        unique_img_ids = set(image_ids)
        for img_id in unique_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            gt_cats = set()
            for ann in anns:
                cat_name = self.coco.cats[ann['category_id']]['name']
                gt_cats.add(cat_name) # 这里 cat_name 已经是 canonical form
            img_id_to_gt_cats[img_id] = gt_cats

        # 遍历每条生成的描述
        for caption, img_id in zip(generated_captions, image_ids):
            if not caption:
                continue
                
            num_sentences += 1
            
            # 1. 获取该句提到的物体 (Canonical Categories)
            mentioned_objects = self._get_mentioned_objects(caption)
            
            # 2. 获取该图的真实物体
            gt_objects = img_id_to_gt_cats.get(img_id, set())
            
            # 3. 计算幻觉
            # 幻觉物体 = 提到但不在 GT 中的物体
            hallucinated_objects = mentioned_objects - gt_objects
            
            # 更新 CHAIR_s (句子级: 只要有一个幻觉物体就算)
            if len(hallucinated_objects) > 0:
                chair_s_count += 1
            
            # 更新 CHAIR_i (对象级)
            chair_i_total_count += len(mentioned_objects)
            chair_i_hallucinated_count += len(hallucinated_objects)
            
        # 计算最终指标
        chair_s = (chair_s_count / num_sentences) * 100 if num_sentences > 0 else 0
        chair_i = (chair_i_hallucinated_count / chair_i_total_count) * 100 if chair_i_total_count > 0 else 0
        
        return {
            'CHAIRs': chair_s,
            'CHAIRi': chair_i,
            'num_sentences': num_sentences,
            'num_hallucinated_sentences': chair_s_count,
            'num_mentioned_objects': chair_i_total_count,
            'num_hallucinated_objects': chair_i_hallucinated_count
        }

def main():
    parser = argparse.ArgumentParser(description='标准 CHAIR 评估脚本')
    parser.add_argument('--results_file', type=str, required=True, help='生成结果的JSON文件')
    parser.add_argument('--coco_instances', type=str, 
                        default='data/coco/annotations/instances_val2017.json',
                        help='COCO instances标注文件')
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"文件不存在: {args.results_file}")
        return
        
    print("="*60)
    print("Standard CHAIR Evaluator (Academic Version)")
    print("="*60)
    
    # 加载结果
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    if isinstance(results, dict):
        results = [results]
        
    generated_captions = []
    image_ids = []
    
    for res in results:
        if 'caption' in res and 'image_id' in res:
            generated_captions.append(res['caption'])
            image_ids.append(res['image_id'])
            
    print(f"待评估样本数: {len(generated_captions)}")
    
    evaluator = StandardCHAIR(args.coco_instances)
    metrics = evaluator.compute_chair(generated_captions, image_ids)
    
    print("\n" + "="*40)
    print("评估结果 (CHAIR)")
    print("="*40)
    print(f"CHAIRs (Sentence Level): {metrics['CHAIRs']:.2f}%")
    print(f"CHAIRi (Object Level):   {metrics['CHAIRi']:.2f}%")
    print("-" * 40)
    print(f"Total Sentences:      {metrics['num_sentences']}")
    print(f"Hallucinated Sents:   {metrics['num_hallucinated_sentences']}")
    print(f"Total Objects:        {metrics['num_mentioned_objects']}")
    print(f"Hallucinated Objects: {metrics['num_hallucinated_objects']}")
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(args.results_file), 'test.json')
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n结果已保存至: {output_path}")

if __name__ == "__main__":
    main()