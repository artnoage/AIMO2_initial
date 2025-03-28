# GRPO Training Scripts for Mathematical Problem Solving

This directory contains scripts for training language models using Generative Reinforcement Policy Optimization (GRPO) for mathematical problem-solving tasks. These training scripts form the improvement component of the larger mathematical problem-solving framework.

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

Each reward function corresponds to a specific benchmark in the `benchmarks` directory and uses the same evaluation criteria to ensure consistency between training and evaluation.

### Dynamic Reward (`dynamic_reward.py`)
A flexible reward function that dynamically selects between different reward types based on the example type:
- Handles multiple example types in a single training run
- Maintains consistent statistics across reward types
- Supports group-based rewards for solution diversity
- Uses `utils.similarity_checker` to measure solution diversity

### Reward Statistics (`reward_stats.py`)
Tracks detailed statistics during training:
- Records reward distributions and components
- Maintains separate statistics for each reward type
- Provides human-readable summaries for logging
- Integrates with Wandb for visualization

### Configuration (`config.py`)
Defines the `RewardConfig` dataclass with settings for:
- Model parameters (main and auxiliary models)
- Reward values for different components
- Embedding model settings for similarity checking
- Numeric tolerance and other execution parameters

## Training Scripts

### Dynamic Training
- `dynamic_qwen0.py`, `dynamic_qwen1.py`, `dynamic_qwen2.py`: Train Qwen models with dynamic rewards using different seeds and configurations
  - Uses combined datasets prepared with `utils.data_preparation.prepare_combined_data`
  - Balances multiple task types in a single training run

### Task-Specific Training
- `solution_qwen0.py`: Train for complete mathematical solutions
  - Corresponds to `benchmarks.standard_benchmark`
  - Uses `utils.data_preparation.prepare_solution_data`

- `programming_qwen0.py`: Train for Python code generation
  - Corresponds to `benchmarks.programming_benchmark`
  - Uses `utils.data_preparation.prepare_programming_data`

- `test_programming_qwen0.py`: Train for test function creation
  - Corresponds to `benchmarks.test_benchmark`
  - Uses `utils.data_preparation.prepare_test_programming_data`

- `tutor_grpo.py`: Train for error identification and correction
  - Corresponds to `benchmarks.tutor_benchmark`
  - Uses `utils.data_preparation.prepare_tutor_data`

- `finalization_grpo.py`: Train for completing partial solutions
  - Uses `utils.data_preparation.prepare_finalization_data`
  - Focuses on completing solutions from partial steps

## Integration with Other Components

### Connection to Benchmarks
Each training script is designed to improve performance on specific benchmarks:
- `solution_qwen0.py` → Improves performance on `benchmarks.standard_benchmark.py`
- `programming_qwen0.py` → Improves performance on `benchmarks.programming_benchmark.py`
- `test_programming_qwen0.py` → Improves performance on `benchmarks.test_benchmark.py`
- `tutor_grpo.py` → Improves performance on `benchmarks.tutor_benchmark.py`

### Utility Dependencies
Training scripts rely on utility modules in the `utils` directory:
- `utils.data_preparation`: Formats datasets for different training tasks
- `utils.solution_utils`: Validates solutions and extracts answers
- `utils.similarity_checker`: Measures diversity between solutions
- `utils.model_utils`: Handles model responses and timeouts

### Dataset Processing
Training uses datasets processed by the auxiliary scripts:
- Filtered datasets from `auxilary.filter_dataset`
- Validation sets from `auxilary.create_validation_set`
- Merged datasets from `auxilary.merge_json`

## Common Features

All training scripts share these common features:
- Integration with Unsloth for efficient training of Qwen models
- Wandb logging for experiment tracking and visualization
- LoRA fine-tuning for parameter-efficient training
- Detailed logging of training metrics and reward components
- Model saving in merged format for easy deployment
- Checkpoint creation for resuming training

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

# Train a model for tutoring and error identification
python -m grpo.tutor_grpo
```

## Model Export and Deployment

After training, models can be exported using `auxilary.export_model.py`:
```bash
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/dynamic_0_20240315 --output-dir models/dynamic_0
```

Exported models can then be evaluated using the benchmark scripts to measure improvement:
```bash
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000
```
