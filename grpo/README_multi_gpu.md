# Multi-GPU Training for GRPO Solver

This directory contains scripts and configurations for training the GRPO solver model using multiple GPUs. The implementation uses the Transformers Reinforcement Learning (TRL) library instead of unsloth, providing support for multi-GPU training and Qwen3 models.

## Files Overview

- `solver_ref_4b_trl.py`: Main training script using TRL instead of unsloth
- `run_multi_gpu_training.sh`: Script to launch training with multiple GPUs using Accelerate
- `run_deepspeed_training.sh`: Script to launch training with DeepSpeed optimization
- `ds_config.json`: DeepSpeed configuration file for optimized multi-GPU training

## Requirements

Before running the training, make sure you have the following packages installed:

```bash
pip install transformers trl accelerate deepspeed
```

## Training Options

### Option 1: Basic Multi-GPU Training with Accelerate

This option uses the Accelerate library to distribute training across multiple GPUs:

```bash
cd grpo
./run_multi_gpu_training.sh --gpus 4
```

Parameters:
- `--gpus`: Number of GPUs to use (default: 4)
- `--script`: Path to the training script (default: solver_ref_4b_trl.py)

### Option 2: Optimized Multi-GPU Training with DeepSpeed

This option uses DeepSpeed for more optimized multi-GPU training with memory efficiency:

```bash
cd grpo
./run_deepspeed_training.sh --gpus 4
```

Parameters:
- `--gpus`: Number of GPUs to use (default: 4)
- `--script`: Path to the training script (default: solver_ref_4b_trl.py)
- `--ds_config`: Path to DeepSpeed config file (default: ds_config.json)

## DeepSpeed Configuration

The DeepSpeed configuration (`ds_config.json`) is set up with:

- ZeRO stage 2 optimization
- CPU offloading for optimizer states
- Mixed precision training (BF16 or FP16 based on hardware support)
- Gradient accumulation and clipping

You can modify the configuration file to adjust these settings based on your hardware and requirements.

## Key Improvements Over the Original Implementation

1. **Multi-GPU Support**: Distributes training across multiple GPUs for faster training
2. **Memory Efficiency**: Uses gradient checkpointing and ZeRO optimization to reduce memory usage
3. **Model Compatibility**: Works with Qwen3 and other models through standard HuggingFace interfaces
4. **Flexible Precision**: Automatically uses BF16 or FP16 based on hardware support
5. **DeepSpeed Integration**: Optional DeepSpeed support for further optimization

## Customizing Training Parameters

You can modify the training parameters in `solver_ref_4b_trl.py`:

- Batch size: Adjust `per_device_train_batch_size` and `gradient_accumulation_steps`
- Learning rate: Modify `learning_rate`
- Training epochs: Change `num_train_epochs`
- Model loading: Update `model_name` to point to your model

## Monitoring Training

The training progress is logged to:
- Console output
- Log files in the `logs/` directory
- Weights & Biases (wandb) if configured

## Saving and Loading Models

The trained model is saved to:
```
models/{model_type}/{timestamp}/
```

You can load the saved model using standard HuggingFace methods:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/saved/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/model")
