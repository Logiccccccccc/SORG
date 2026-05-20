# Vision Caption Correction

This project provides a vision-reasoning-enhanced image captioning pipeline. It uses a VLM such as LLaVA together with panoptic segmentation, CLIP node features, spatial relations, a causal GNN, and uncertainty-aware object inference.

## Project structure

```text
vision-caption-correction/
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   ├── infer_llava.yaml
│   └── infer_qwen.yaml
├── eval/
│   ├── evaluate_cpr_poc.py
│   ├── evaluate_spice_fixed.py
│   ├── evaluation_CHAIR.py
│   └── evaluation_pope.py
├── sorg/
│   ├── config.py
│   ├── caption_generator.py
│   ├── data_preparation.py
│   ├── inference.py
│   ├── knowledge_builder.py
│   ├── model_core.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Configuration


- `configs/default.yaml`: common paths, model settings, GNN parameters, training defaults, and inference thresholds.
- `configs/train.yaml`: training overrides.
- `configs/infer_llava.yaml`: LLaVA inference overrides.
- `configs/infer_qwen.yaml`: Qwen inference overrides.

All paths in YAML are relative to the project root, for example:

```yaml
COCO_VAL_IMG: data/coco/val2017
CHECKPOINT_DIR: checkpoints
LLAVA_MODEL_PATH: models/llava-1.5-7b-hf
```

At runtime, `sorg/config.py` resolves these relative paths against the project root.

## Data and model layout

Prepare the project like this:

```text
vision-caption-correction/
├── data/
│   └── coco/
│       ├── train2017/
│       ├── val2017/
│       └── annotations/
│           ├── instances_train2017.json
│           ├── instances_val2017.json
│           ├── captions_train2017.json
│           └── captions_val2017.json
├── models/
│   └── llava-1.5-7b-hf/
├── checkpoints/
├── knowledge_base/
└── results/
```

You can change these locations in the YAML files if your server uses a different directory layout.

## Installation

Create a Python environment first, then install the project dependencies from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If the PyTorch version in `requirements.txt` does not match your server CUDA version, install the correct PyTorch build for your CUDA environment first, then install the remaining dependencies from `requirements.txt`.

SPICE evaluation requires Java.

## Training with LLaVA settings

Training uses `configs/train.yaml` by default:

```bash
python sorg/train.py \
  --config configs/train.yaml \
  --max_train_samples 10000 \
  --max_val_samples 1000 \
  --num_epochs 30 \
  --target_mode binary
```

Useful options:

```bash
python sorg/train.py \
  --config configs/train.yaml \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --resume checkpoints/new_gnn_epoch_10.pt
```

Training outputs are saved to:

- `checkpoints/gnn_state_5000.pt`
- `checkpoints/new_gnn_epoch_*.pt`
- `knowledge_base/`

## Inference with LLaVA

Single-image inference:

```bash
python sorg/inference.py \
  --config configs/infer_llava.yaml \
  --mode single \
  --image_path data/coco/val2017/000000000139.jpg \
  --output_file results/llava_single.json \
  --ablation_mode complete
```

Batch inference:

```bash
python sorg/inference.py \
  --config configs/infer_llava.yaml \
  --mode batch \
  --annotation_file data/coco/annotations/instances_val2017.json \
  --image_dir data/coco/val2017 \
  --output_file results/llava_batch.json \
  --max_samples 500 \
  --ablation_mode complete
```

Supported ablation modes:

- `complete`: full method.
- `baseline`: VLM only, without GNN reasoning.
- `no_reasoning`: skips causal reasoning.
- `no_relations`: disables spatial relations.
- `no_uncertainty`: disables uncertainty estimation.
- `no_clip`: replaces CLIP features with random features.

## Evaluation

### CHAIR

```bash
python eval/evaluation_CHAIR.py \
  --results_file results/llava_batch.json \
  --coco_instances data/coco/annotations/instances_val2017.json
```

### SPICE

```bash
python eval/evaluate_spice_fixed.py \
  --results_file results/llava_batch.json \
  --coco_captions_file data/coco/annotations/captions_val2017.json \
  --min_words 4
```

### CPR and PoC

```bash
python eval/evaluate_cpr_poc.py \
  --model_file results/llava_batch.json \
  --baseline_file results/llava_baseline.json \
  --annotation_file data/coco/annotations/instances_val2017.json \
  --output_file results/cpr_poc.json
```

### POPE

```bash
python eval/evaluation_pope.py \
  --config configs/infer_llava.yaml \
  --image_dir data/coco/val2017 \
  --coco_instances data/coco/annotations/instances_val2017.json \
  --num_images 500 \
  --output_file results/pope_llava.json \
  --ablation_mode complete
```

## Qwen inference

Use `configs/infer_qwen.yaml`:

```bash
python sorg/inference.py \
  --config configs/infer_qwen.yaml \
  --mode batch \
  --annotation_file data/coco/annotations/instances_val2017.json \
  --image_dir data/coco/val2017 \
  --output_file results/qwen_batch.json \
  --max_samples 500 \
  --ablation_mode complete
```

