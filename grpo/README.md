# GRPO Training Scripts for Mathematical Problem Solving

This directory contains scripts for training language models using Generative Reinforcement Policy Optimization (GRPO) for mathematical problem-solving tasks.

## Core Components

### Reward Functions (`rewards.py`)
Defines various reward functions for different mathematical problem-solving tasks:
- `BaseReward`: Abstract base class for all reward functions
- `SolutionReward`: Rewards for complete mathematical solutions
- `FinalizationReward`: Rewards for completing partial solutions
- `ProgrammingReward`: Rewards for Python code that solves math problems
- `TutorReward`: Rewards for identifying errors in solutions
- `TestProgrammingReward`: Rewards for creating test functions
- `ArchitectReward`: Rewards for creating solution architectures

### Dynamic Reward (`dynamic_reward.py`)
A flexible reward function that dynamically selects between different reward types based on the example type:
- Handles multiple example types in a single training run
- Maintains consistent statistics across reward types
- Supports group-based rewards for solution diversity

### Reward Statistics (`reward_stats.py`)
Tracks detailed statistics during training:
- Records reward distributions and components
- Maintains separate statistics for each reward type
- Provides human-readable summaries for logging

### Configuration (`config.py`)
Defines the `RewardConfig` dataclass with settings for:
- Model parameters (main and auxiliary models)
- Reward values for different components
- Embedding model settings for similarity checking
- Numeric tolerance and other execution parameters

## Training Scripts

### Dynamic Training
- `dynamic_qwen0.py`, `dynamic_qwen1.py`, `dynamic_qwen2.py`: Train Qwen models with dynamic rewards using different seeds and configurations

### Task-Specific Training
- `solution_qwen0.py`: Train for complete mathematical solutions
- `programming_qwen0.py`: Train for Python code generation
- `test_programming_qwen0.py`: Train for test function creation
- `tutor_grpo.py`: Train for error identification and correction
- `finalization_grpo.py`: Train for completing partial solutions

## Common Features

All training scripts share these common features:
- Integration with Unsloth for efficient training
- Wandb logging for experiment tracking
- LoRA fine-tuning for parameter-efficient training
- Detailed logging of training metrics
- Model saving in merged format

## Usage

Run any training script directly:

```bash
python -m grpo.dynamic_qwen0
```

## Configuration

Training scripts use the `RewardConfig` class from `config.py` for configuration. Key parameters include:

### Model Settings
- `model_type`: Type of model to train (e.g., "dynamic_0", "solution", "programming")
- Base reward values for different components
- Similarity thresholds and diversity bonuses

### Training Settings
Each script configures:
- Learning rate and optimizer settings
- Batch size and gradient accumulation steps
- Number of generations per prompt
- Maximum sequence lengths
- Training epochs and checkpointing frequency

## Example Usage

```bash
# Train a dynamic model with multiple task types
python -m grpo.dynamic_qwen0

# Train a model specifically for programming tasks
python -m grpo.programming_qwen0
```
