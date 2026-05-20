import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SORG_DIR = os.path.join(PROJECT_ROOT, 'sorg')
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SORG_DIR not in sys.path:
    sys.path.insert(0, SORG_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
import json
import argparse
import re
import zipfile
import nltk
from collections import defaultdict
from pycocotools.coco import COCO
from config import Config
os.environ.setdefault('NLTK_DATA', Config.NLTK_DATA_DIR)
os.makedirs(os.environ['NLTK_DATA'], exist_ok=True)

def _ensure_nltk_resource(resource_path: str, package: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, download_dir=os.environ['NLTK_DATA'], quiet=True)
_ensure_nltk_resource('tokenizers/punkt', 'punkt')
_ensure_nltk_resource('corpora/wordnet', 'wordnet')
_ensure_nltk_resource('corpora/omw-1.4', 'omw-1.4')
from nltk.stem import WordNetLemmatizer
COCO_SYNONYMS = {'airplane': ['airplane', 'aeroplane', 'plane', 'jet', '747', 'airplanes', 'aeroplanes', 'planes', 'jets', '747s'], 'apple': ['apple', 'apples'], 'backpack': ['backpack', 'backpacks', 'rucksack', 'rucksacks', 'satchel', 'satchels', 'pack', 'packs'], 'banana': ['banana', 'bananas'], 'baseball bat': ['baseball bat', 'baseball bats', 'bat', 'bats'], 'baseball glove': ['baseball glove', 'baseball gloves', 'glove', 'gloves', 'mitt', 'mitts'], 'bear': ['bear', 'bears', 'teddy bear', 'teddy bears', 'teddy', 'teddies'], 'book': ['book', 'books'], 'bottle': ['bottle', 'bottles', 'water bottle', 'water bottles', 'pop bottle', 'pop bottles'], 'car': ['car', 'cars', 'auto', 'autos', 'automobile', 'automobiles', 'sedan', 'sedans', 'suv', 'suvs', 'jeep', 'jeeps', 'taxi', 'taxis', 'cab', 'cabs', 'van', 'vans', 'minivan', 'minivans'], 'carrot': ['carrot', 'carrots'], 'cat': ['cat', 'cats', 'kitten', 'kittens', 'kitty', 'kitties', 'feline', 'felines'], 'cell phone': ['cell phone', 'cell phones', 'mobile phone', 'mobile phones', 'cellphone', 'cellphones', 'smartphone', 'smartphones', 'phone', 'phones'], 'chair': ['chair', 'chairs', 'seat', 'seats', 'stool', 'stools'], 'elephant': ['elephant', 'elephants'], 'fire hydrant': ['fire hydrant', 'fire hydrants', 'hydrant', 'hydrants'], 'fork': ['fork', 'forks'], 'frisbee': ['frisbee', 'frisbees', 'disk', 'disks'], 'giraffe': ['giraffe', 'giraffes'], 'hair drier': ['hair drier', 'hair driers', 'hair dryer', 'hair dryers', 'blow dryer', 'blow dryers'], 'handbag': ['handbag', 'handbags', 'purse', 'purses', 'bag', 'bags', 'pocketbook', 'pocketbooks'], 'horse': ['horse', 'horses', 'pony', 'ponies', 'foal', 'foals', 'mare', 'mares', 'stallion', 'stallions'], 'hot dog': ['hot dog', 'hot dogs', 'hotdog', 'hotdogs', 'frank', 'franks', 'frankfurter', 'frankfurters'], 'keyboard': ['keyboard', 'keyboards', 'key board', 'key boards'], 'kite': ['kite', 'kites'], 'parking meter': ['parking meter', 'parking meters'], 'person': ['person', 'people', 'persons', 'man', 'men', 'woman', 'women', 'boy', 'boys', 'girl', 'girls', 'child', 'children', 'kid', 'kids', 'guy', 'guys', 'lady', 'ladies', 'gentleman', 'gentlemen', 'baby', 'babies', 'toddler', 'toddlers', 'player', 'players', 'spectator', 'spectators', 'pedestrian', 'pedestrians', 'policeman', 'policemen', 'worker', 'workers'], 'pizza': ['pizza', 'pizzas', 'pie', 'pies'], 'potted plant': ['potted plant', 'potted plants', 'plant', 'plants', 'houseplant', 'houseplants', 'flower', 'flowers', 'pot', 'pots', 'vase', 'vases', 'bush', 'bushes', 'shrub', 'shrubs', 'tree', 'trees'], 'refrigerator': ['refrigerator', 'refrigerators', 'fridge', 'fridges'], 'remote': ['remote', 'remotes', 'remote control', 'remote controls', 'controller', 'controllers'], 'sandwich': ['sandwich', 'sandwiches', 'sub', 'subs', 'burger', 'burgers', 'hamburger'], 'scissors': ['scissors', 'shears', 'clippers'], 'suitcase': ['suitcase', 'suitcases', 'luggage', 'cases', 'trunk', 'trunks', 'valise', 'valises'], 'surfboard': ['surfboard', 'surfboards'], 'teddy bear': ['teddy bear', 'teddy bears', 'teddy', 'teddies', 'stuffed animal', 'stuffed animals', 'plush toy', 'plush toys'], 'tennis racket': ['tennis racket', 'tennis rackets', 'racket', 'rackets', 'racquet', 'racquets'], 'tie': ['tie', 'ties', 'necktie', 'neckties', 'bowtie', 'bowties'], 'toaster': ['toaster', 'toasters'], 'toilet': ['toilet', 'toilets', 'commode', 'commodes', 'lavatory', 'lavatories', 'potty', 'potties'], 'umbrella': ['umbrella', 'umbrellas', 'parasol', 'parasols'], 'wine glass': ['wine glass', 'wine glasses', 'wineglass', 'wineglasses', 'glass', 'glasses', 'goblet', 'goblets'], 'zebra': ['zebra', 'zebras']}

class StandardCHAIR:

    def __init__(self, coco_instances_file):
        print(f'Loading instances annotation: {coco_instances_file}')
        import json
        with open(coco_instances_file, 'r') as f:
            full_data = json.load(f)
        object_anns = [ann for ann in full_data.get('annotations', []) if 'category_id' in ann]
        full_data['annotations'] = object_anns
        temp_instances_file = coco_instances_file.replace('.json', '_instances_only.json')
        with open(temp_instances_file, 'w') as f:
            json.dump(full_data, f)
        self.coco = COCO(temp_instances_file)
        self.lemmatizer = WordNetLemmatizer()
        self.word_to_cat = {}
        for cat, synonyms in COCO_SYNONYMS.items():
            for syn in synonyms:
                self.word_to_cat[syn.lower()] = cat
        self.cat_id_to_name = {}
        self.cat_name_to_id = {}
        for cat_id, cat_info in self.coco.cats.items():
            cat_name = cat_info['name']
            self.cat_id_to_name[cat_id] = cat_name
            self.cat_name_to_id[cat_name] = cat_id
        print(f'Synonym dictionary built with {len(self.word_to_cat)} word variants.')
        self._non_noun_followers = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'and', 'or', 'but', 'on', 'in', 'at', 'with', 'of', 'to', 'from', 'near', 'next', 'beside', 'behind', 'under', 'over', 'above'}

    def _tokenize(self, caption):
        return nltk.word_tokenize(caption.lower())

    def _get_mentioned_objects(self, caption):
        tokens = self._tokenize(caption)
        mentioned_cats = set()
        skip_indices = set()
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                bi_gram = f'{tokens[i]} {tokens[i + 1]}'
                if bi_gram in self.word_to_cat:
                    mentioned_cats.add(self.word_to_cat[bi_gram])
                    skip_indices.add(i)
                    skip_indices.add(i + 1)
        for i, token in enumerate(tokens):
            if i in skip_indices:
                continue
            if token in self.word_to_cat:
                mentioned_cats.add(self.word_to_cat[token])
                continue
            lemma = self.lemmatizer.lemmatize(token, pos='n')
            if lemma in self.word_to_cat:
                mentioned_cats.add(self.word_to_cat[lemma])
        if not mentioned_cats:
            return set()
        if 'orange' in mentioned_cats:
            is_color = False
            for i, tok in enumerate(tokens):
                if tok == 'orange':
                    nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                    if nxt and nxt not in self._non_noun_followers:
                        is_color = True
                        break
                    if nxt == 'and':
                        nxt2 = tokens[i + 2] if i + 2 < len(tokens) else None
                        if nxt2 and nxt2 not in self._non_noun_followers:
                            is_color = True
                            break
            if is_color:
                mentioned_cats.discard('orange')
        caption_lower = caption.lower()
        final_cats = set(mentioned_cats)
        if 'no' in tokens:
            for cat in list(final_cats):
                candidates = [k for k, v in self.word_to_cat.items() if v == cat]
                for cand in candidates:
                    if re.search(f'\\bno\\s+{re.escape(cand)}\\b', caption_lower):
                        final_cats.discard(cat)
                        break
        if 'sign' in tokens or 'signs' in tokens:
            pass
        return final_cats

    def compute_chair(self, generated_captions, image_ids):
        chair_s_count = 0
        chair_i_hallucinated_count = 0
        chair_i_total_count = 0
        num_sentences = 0
        img_id_to_gt_cats = {}
        unique_img_ids = set(image_ids)
        for img_id in unique_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            gt_cats = set()
            for ann in anns:
                cat_name = self.coco.cats[ann['category_id']]['name']
                gt_cats.add(cat_name)
            img_id_to_gt_cats[img_id] = gt_cats
        for caption, img_id in zip(generated_captions, image_ids):
            if not caption:
                continue
            num_sentences += 1
            mentioned_objects = self._get_mentioned_objects(caption)
            gt_objects = img_id_to_gt_cats.get(img_id, set())
            hallucinated_objects = mentioned_objects - gt_objects
            if len(hallucinated_objects) > 0:
                chair_s_count += 1
            chair_i_total_count += len(mentioned_objects)
            chair_i_hallucinated_count += len(hallucinated_objects)
        chair_s = chair_s_count / num_sentences * 100 if num_sentences > 0 else 0
        chair_i = chair_i_hallucinated_count / chair_i_total_count * 100 if chair_i_total_count > 0 else 0
        return {'CHAIRs': chair_s, 'CHAIRi': chair_i, 'num_sentences': num_sentences, 'num_hallucinated_sentences': chair_s_count, 'num_mentioned_objects': chair_i_total_count, 'num_hallucinated_objects': chair_i_hallucinated_count}

def main():
    parser = argparse.ArgumentParser(description='Standard CHAIR evaluation script')
    parser.add_argument('--results_file', type=str, required=True, help='JSON file with generated results.')
    parser.add_argument('--coco_instances', type=str, default='data/coco/annotations/instances_val2017.json', help='COCO instances annotation file.')
    args = parser.parse_args()
    if not os.path.exists(args.results_file):
        print(f'File does not exist: {args.results_file}')
        return
    print('=' * 60)
    print('Standard CHAIR Evaluator (Academic Version)')
    print('=' * 60)
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
    print(f'Number of samples to evaluate: {len(generated_captions)}')
    evaluator = StandardCHAIR(args.coco_instances)
    metrics = evaluator.compute_chair(generated_captions, image_ids)
    print('\n' + '=' * 40)
    print('Evaluation Results (CHAIR)')
    print('=' * 40)
    print(f"CHAIRs (Sentence Level): {metrics['CHAIRs']:.2f}%")
    print(f"CHAIRi (Object Level):   {metrics['CHAIRi']:.2f}%")
    print('-' * 40)
    print(f"Total Sentences:      {metrics['num_sentences']}")
    print(f"Hallucinated Sents:   {metrics['num_hallucinated_sentences']}")
    print(f"Total Objects:        {metrics['num_mentioned_objects']}")
    print(f"Hallucinated Objects: {metrics['num_hallucinated_objects']}")
    output_path = os.path.join(os.path.dirname(args.results_file), 'test.json')
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nResults saved to: {output_path}')
if __name__ == '__main__':
    main()
