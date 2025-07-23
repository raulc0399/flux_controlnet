# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FLUX model fine-tuning project that uses LoRA (Low-Rank Adaptation) to train custom image generation models. The project is based on the Oxen.ai FLUX fine-tuning implementation and uses Marimo for an interactive notebook interface.

## Key Architecture Components

### Core Training Pipeline
- **Model**: FLUX.1-dev transformer model with LoRA adapters
- **Data**: Custom dataset with images and captions loaded from Oxen.ai repositories
- **Training**: Flow matching loss with rectified flow formulation
- **Optimization**: AdamW8bit optimizer with gradient clipping and constant learning rate
- **Sampling**: FlowMatchEulerDiscreteScheduler with dynamic shifting for high-resolution images

### Model Components (load_models function - marimo_train.py:671)
- FluxTransformer2DModel with LoRA applied to attention and MLP layers
- AutoencoderKL for latent space encoding/decoding
- Dual text encoders: CLIP and T5 for comprehensive text understanding
- Corresponding tokenizers for text processing

### Training Configuration (marimo_train.py:108-144)
- LoRA rank/alpha: 16/16 for efficient fine-tuning
- Batch size: 1 with gradient accumulation
- Learning rate: 1e-4 with AdamW8bit optimizer
- Training steps: 2000 with checkpoints every 200 steps
- Multi-resolution training: 512/768/1024px with aspect ratio preservation

## Commands

### Running Training
```bash
python marimo_train.py run train --model-name "black-forest-labs/FLUX.1-dev" --repo-name "your-repo" --dataset-file "train.parquet" --images-directory "images" --hf-api-key "your-key"
```

### Interactive Notebook
```bash
marimo run marimo_train.py
```

## Development Notes

### Memory Management
- Uses gradient checkpointing and memory flushing (flush_memory function)
- bfloat16 precision for training with float32 for sampling
- Automatic GPU memory cleanup between operations

### Data Requirements
- Parquet dataset with 'image' and 'action' columns
- Images directory with corresponding image files
- Images automatically resized to multiples of 16 for FLUX compatibility
- Optional trigger phrase for concept training

### Output Structure
- Models saved as SafeTensors format
- Training logs in JSONL format
- Sample images generated during training
- Results uploaded to Oxen.ai with automatic branch management

### Key Functions
- `train()`: Main training loop with flow matching loss
- `generate_samples()`: Inference during training for monitoring
- `FluxDataset`: Custom dataset class with multi-resolution support
- `load_models()`: Model loading with LoRA configuration
- `write_and_save_results()`: Checkpoint saving and uploading