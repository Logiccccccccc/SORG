# Vision Caption Correction

## Overview

This project implements **SORG (Structured Object Relational Grounding)**, a vision-reasoning-enhanced image captioning framework designed to reduce object hallucination in multimodal large language models (MLLMs). Object hallucination refers to the problem where a generated caption mentions objects that are not actually supported by the input image. This issue is especially important for image captioning, because a single unsupported object noun can make an otherwise fluent caption factually unreliable.

SORG introduces a lightweight structured reasoning stage between visual perception and caption generation. Instead of directly prompting a vision-language model with raw image input alone, the system first detects candidate objects, builds an object-level scene representation, estimates relational support among objects, and then converts the inferred scene-supported object set into prompt-level guidance for a frozen VLM such as **LLaVA-1.5**, **Qwen-VL**, **MiniGPT-4 / BLIP-2**, or **InstructBLIP**.

The core idea is that an object should not only be visually plausible in isolation, but also supported by the broader scene context. For example, SORG uses detected objects, object confidence scores, spatial relations, CLIP-based semantic node features, and graph-based relational reasoning to determine which objects are reliable enough to guide caption generation. This helps suppress unsupported object mentions while preserving valid visual content and allowing the VLM to generate natural captions.

The pipeline contains three main stages:

1. **Perception and graph construction**  
   The image is processed by MaskFormer to obtain candidate object regions, labels, confidence scores, and bounding boxes. Each object is represented using a fusion of visual crop features, text label features, and positional information. A sparse object graph is then constructed using heuristic spatial relations.

2. **Structured relational inference**  
   A lightweight CausalGNN estimates object co-occurrence and directional dependency scores. The system first selects visually reliable core objects and then infers additional plausible objects based on their relational support from the core object set.

3. **Scene-supported caption generation**  
   The final scene-supported object set is converted into a structured prompt. The frozen VLM is instructed to mention only supported object nouns while remaining free to describe valid attributes, actions, and spatial relations. This is a soft prompt-level constraint rather than a hard decoding mask.

The project supports full-model inference, backbone-only baselines, and ablation settings such as disabling relational reasoning, spatial relations, uncertainty estimation, or CLIP semantic features. It also provides evaluation scripts for hallucination-focused metrics such as **CHAIR**, **POPE**, **CPR**, and **PoC**, as well as caption-quality metrics such as **SPICE**.

## Project Structure

```text
.
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   ├── infer_llava.yaml
│   └── infer_qwen.yaml
├── sorg/
│   ├── config.py
│   ├── train.py
│   ├── inference.py
│   ├── caption_generator.py
│   ├── data_preparation.py
│   ├── knowledge_builder.py
│   └── model_core.py
├── eval/
│   ├── evaluation_CHAIR.py
│   ├── evaluation_pope.py
│   ├── evaluate_cpr_poc.py
│   └── evaluate_spice_fixed.py
├── data/
├── models/
├── checkpoints/
├── knowledge_base/
├── results/
├── requirements.txt
└── README.md
```

## Installation

Create a Python environment and install the project dependencies from the provided requirements file:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place COCO data under the project directory using the following structure:

```text
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── captions_train2017.json
    └── captions_val2017.json
```

The project uses relative paths by default. You can modify dataset, model, checkpoint, and result paths in the YAML files under `configs/`.

## Model Preparation

By default, the LLaVA configuration expects the model checkpoint at:

```text
models/llava-1.5-7b-hf
```

For Qwen inference, the default path is:

```text
models/qwen3-vl-8b-instruct-abliterated-CV
```

You can change these paths in:

```text
configs/infer_llava.yaml
configs/infer_qwen.yaml
```

If a model path points to a local directory, make sure the directory contains the required Hugging Face model files. If it points to a remote model ID, the Transformers loader may download it automatically, depending on your server environment.

## Configuration System

The original monolithic configuration has been decoupled into YAML files:

- `configs/default.yaml`: shared default configuration
- `configs/train.yaml`: training-specific configuration
- `configs/infer_llava.yaml`: LLaVA inference configuration
- `configs/infer_qwen.yaml`: Qwen inference configuration

The Python scripts load these YAML files through `sorg/config.py`. All relative paths are resolved from the project root.

## Training with LLaVA

Training builds the object vocabulary, extracts co-occurrence and directional relation statistics, trains the Fusion MLP and CausalGNN, and saves the knowledge base and model checkpoint.

Example:

```bash
python sorg/train.py \
    --config configs/train.yaml \
    --max_train_samples 10000 \
    --max_val_samples 1000 \
    --num_epochs 30 \
    --target_mode binary
```

Main outputs:

```text
checkpoints/gnn_state_5000.pt
knowledge_base/object_vocab.json
knowledge_base/statistics.json
knowledge_base/cooccurrence_matrix.npy
knowledge_base/causality_matrix.npy
```

You can resume training from a checkpoint:

```bash
python sorg/train.py \
    --config configs/train.yaml \
    --resume checkpoints/new_gnn_epoch_10.pt \
    --num_epochs 30
```

## Inference with LLaVA

### Single-image inference

```bash
python sorg/inference.py \
    --config configs/infer_llava.yaml \
    --mode single \
    --image_path data/coco/val2017/000000000139.jpg \
    --output_file results/single_llava.json \
    --ablation_mode complete
```

### Batch inference

```bash
python sorg/inference.py \
    --config configs/infer_llava.yaml \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/llava_complete.json \
    --max_samples 500 \
    --ablation_mode complete
```

### Baseline inference

This runs the VLM backbone without SORG reasoning:

```bash
python sorg/inference.py \
    --config configs/infer_llava.yaml \
    --mode batch \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/llava_baseline.json \
    --max_samples 500 \
    --ablation_mode baseline
```

## Ablation Modes

The inference and POPE evaluation scripts support the following modes:

| Mode | Description |
| --- | --- |
| `complete` | Full SORG pipeline |
| `baseline` | Backbone VLM only |
| `no_reasoning` | Uses visual detection but skips GNN reasoning |
| `no_relations` | Removes spatial relation edges |
| `no_uncertainty` | Disables MC-dropout uncertainty estimation |
| `no_clip` | Replaces CLIP node features with random features |

## Evaluation

### CHAIR

CHAIR measures object hallucination at sentence and object-instance levels.

```bash
python eval/evaluation_CHAIR.py \
    --results_file results/llava_complete.json \
    --coco_instances data/coco/annotations/instances_val2017.json
```

### POPE

POPE evaluates object-presence consistency under random, popular, and adversarial negative sampling.

```bash
python eval/evaluation_pope.py \
    --image_dir data/coco/val2017 \
    --coco_instances data/coco/annotations/instances_val2017.json \
    --num_images 500 \
    --model_checkpoint checkpoints/gnn_state_5000.pt \
    --knowledge_base knowledge_base \
    --output_file results/pope_llava_complete.json \
    --ablation_mode complete
```

### CPR and PoC

CPR evaluates how well the corrected caption preserves valid objects from the baseline. PoC evaluates whether object-level additions or deletions are beneficial.

```bash
python eval/evaluate_cpr_poc.py \
    --model_file results/llava_complete.json \
    --baseline_file results/llava_baseline.json \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --output_file results/cpr_poc_llava.json
```

### SPICE

```bash
python eval/evaluate_spice_fixed.py \
    --results_file results/llava_complete.json \
    --coco_captions_file data/coco/annotations/captions_val2017.json \
    --min_words 4
```

SPICE requires Java.

## Qwen Inference

To use Qwen instead of LLaVA:

```bash
python sorg/inference.py \
    --config configs/infer_qwen.yaml \
    --mode batch \
    --model_type qwen \
    --annotation_file data/coco/annotations/instances_val2017.json \
    --image_dir data/coco/val2017 \
    --output_file results/qwen_complete.json \
    --max_samples 500 \
    --ablation_mode complete
```

## Notes

- The MLLM backbone is frozen during SORG training.
- The trained components are lightweight relational modules, mainly the Fusion MLP and CausalGNN.
- The generated visual context is used as prompt-level guidance, not as a hard vocabulary mask.
- Results depend on the quality of object detection, the coverage of the COCO-trained relational knowledge base, and the behavior of the selected VLM backbone.
- All paths in the default configuration are relative to the project root, making the project easier to run on a server.
