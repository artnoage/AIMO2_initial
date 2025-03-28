# GRPO Training Scripts for Mathematical Problem Solving

This directory contains scripts for training language models using Generative Reinforcement Policy Optimization (GRPO) for mathematical problem-solving tasks. These training scripts form the improvement component of the larger mathematical problem-solving framework.

## Core Components

### Reward Functions (`rewards.py`)
Defines various reward functions for different mathematical problem-solving tasks:
- `BaseReward`: Abstract base class for all reward functions with common interface
  - Defines the contract for all reward implementations
  - Handles batch processing with async execution
  - Provides error handling and timeout management

- `SolutionReward`: Rewards for complete mathematical solutions
  - Evaluates correctness of final answers using numeric verification
  - Assesses solution quality, step coherence, and explanation clarity
  - Penalizes common errors like skipped steps or incorrect reasoning

- `FinalizationReward`: Rewards for completing partial solutions
  - Evaluates ability to continue from a given partial solution
  - Checks consistency with provided steps
  - Verifies final answer correctness

- `ProgrammingReward`: Rewards for Python code that solves math problems
  - Executes code in a secure sandbox with timeout protection
  - Verifies output against expected answers with configurable tolerance
  - Evaluates code quality, efficiency, and readability

- `TutorReward`: Rewards for identifying errors in solutions
  - Assesses accuracy of error identification
  - Evaluates quality of explanations and corrections
  - Rewards constructive feedback style

- `TestProgrammingReward`: Rewards for creating test functions
  - Evaluates test function correctness on multiple test cases
  - Assesses robustness to edge cases and numerical precision
  - Rewards comprehensive test coverage

- `ArchitectReward`: Rewards for creating solution architectures
  - Evaluates problem decomposition and approach planning
  - Assesses clarity of guidance for implementation
  - Rewards identification of potential pitfalls and edge cases

Each reward function corresponds to a specific benchmark in the `benchmarks` directory and uses the same evaluation criteria to ensure consistency between training and evaluation.

### Dynamic Reward (`dynamic_reward.py`)
A flexible reward function that dynamically selects between different reward types based on the example type:
- Handles multiple example types in a single training run with type-specific processing
- Maintains consistent statistics across reward types with separate tracking
- Supports group-based rewards for solution diversity using embedding similarity
- Uses `utils.similarity_checker` to measure solution diversity with configurable thresholds
- Implements weighted combinations of reward components based on task type
- Provides detailed component breakdowns for analysis and debugging

### Reward Statistics (`reward_stats.py`)
Tracks detailed statistics during training:
- Records reward distributions and components with histograms and running averages
- Maintains separate statistics for each reward type with task-specific metrics
- Provides human-readable summaries for logging with component breakdowns
- Integrates with Wandb for visualization with custom charts and tables
- Tracks training progress with moving averages and improvement metrics
- Identifies reward outliers for analysis of exceptional cases

### Configuration (`config.py`)
Defines the `RewardConfig` dataclass with settings for:
- Model parameters (main and auxiliary models) with type and endpoint configuration
- Reward values for different components with configurable weights
- Embedding model settings for similarity checking with model selection and parameters
- Numeric tolerance and other execution parameters for consistent evaluation
- Timeout settings for model calls and code execution
- Logging configuration for detailed or summarized output

## Training Scripts

### Dynamic Training
- `dynamic_qwen0.py`, `dynamic_qwen1.py`, `dynamic_qwen2.py`: Train Qwen models with dynamic rewards using different seeds and configurations
  - Uses combined datasets prepared with `utils.data_preparation.prepare_combined_data`
  - Balances multiple task types in a single training run with configurable distribution
  - Implements curriculum learning with increasing difficulty
  - Supports mixed precision training for efficiency
  - Includes evaluation on validation sets during training

### Task-Specific Training
- `solution_qwen0.py`: Train for complete mathematical solutions
  - Corresponds to `benchmarks.standard_benchmark`
  - Uses `utils.data_preparation.prepare_solution_data`
  - Focuses on step-by-step reasoning and answer correctness
  - Implements specialized rewards for mathematical notation quality

- `programming_qwen0.py`: Train for Python code generation
  - Corresponds to `benchmarks.programming_benchmark`
  - Uses `utils.data_preparation.prepare_programming_data`
  - Includes code execution verification in the training loop
  - Rewards efficient and readable code implementations

- `test_programming_qwen0.py`: Train for test function creation
  - Corresponds to `benchmarks.test_benchmark`
  - Uses `utils.data_preparation.prepare_test_programming_data`
  - Focuses on comprehensive test case generation
  - Rewards robust handling of edge cases

- `tutor_grpo.py`: Train for error identification and correction
  - Corresponds to `benchmarks.tutor_benchmark`
  - Uses `utils.data_preparation.prepare_tutor_data`
  - Trains on both correct and deliberately flawed solutions
  - Rewards constructive feedback and clear explanations

- `finalization_grpo.py`: Train for completing partial solutions
  - Uses `utils.data_preparation.prepare_finalization_data`
  - Focuses on completing solutions from partial steps
  - Rewards consistency with provided partial solutions
  - Implements specialized handling for different completion points

## Integration with Other Components

### Connection to Benchmarks
Each training script is designed to improve performance on specific benchmarks:
- `solution_qwen0.py` → Improves performance on `benchmarks.standard_benchmark.py`
- `programming_qwen0.py` → Improves performance on `benchmarks.programming_benchmark.py`
- `test_programming_qwen0.py` → Improves performance on `benchmarks.test_benchmark.py`
- `tutor_grpo.py` → Improves performance on `benchmarks.tutor_benchmark.py`
- `finalization_grpo.py` → Supports multiple benchmarks with partial solution completion

### Utility Dependencies
Training scripts rely on utility modules in the `utils` directory:
- `utils.data_preparation`: Formats datasets for different training tasks with task-specific processing
- `utils.solution_utils`: Validates solutions and extracts answers with LaTeX support
- `utils.similarity_checker`: Measures diversity between solutions using embedding models
- `utils.model_utils`: Handles model responses and timeouts with retry logic

### Dataset Processing
Training uses datasets processed by the auxiliary scripts:
- Filtered datasets from `auxilary.filter_dataset` with quality controls
- Validation sets from `auxilary.create_validation_set` for consistent evaluation
- Merged datasets from `auxilary.merge_json` for comprehensive training
- Converted datasets from `auxilary.datatype_transformation` for format compatibility

## Common Features

All training scripts share these common features:
- Integration with Unsloth for efficient training of Qwen models (2-3x speedup)
- Wandb logging for experiment tracking and visualization with detailed metrics
- LoRA fine-tuning for parameter-efficient training with configurable parameters
- Detailed logging of training metrics and reward components with component breakdowns
- Model saving in merged format for easy deployment with adapter integration
- Checkpoint creation for resuming training with automatic versioning
- Gradient accumulation for effective batch size scaling
- Mixed precision training for memory efficiency
- Evaluation during training on validation sets
- Early stopping based on validation performance

## Usage

Run any training script directly with optional command-line arguments:

```bash
python -m grpo.dynamic_qwen0 --learning_rate 2e-5 --epochs 3 --batch_size 4
```

## Configuration

Training scripts use the `RewardConfig` class from `config.py` for configuration. Key parameters include:

### Model Settings
- `model_type`: Type of model to train (e.g., "dynamic_0", "solution", "programming")
- `model_name`: Base model to fine-tune (e.g., "unsloth/Qwen1.5-7B")
- `auxiliary_model`: Model for reward calculation (e.g., "gpt-4-turbo")
- Base reward values for different components with configurable weights
- Similarity thresholds and diversity bonuses for group-based rewards
- Embedding model configuration for similarity calculation

### Training Settings
Each script configures:
- Learning rate and optimizer settings (AdamW with weight decay)
- Batch size and gradient accumulation steps for memory efficiency
- Number of generations per prompt for diverse training signals
- Maximum sequence lengths for input and output
- Training epochs and checkpointing frequency
- LoRA parameters (rank, alpha, dropout) for adapter configuration
- Evaluation frequency and criteria

## Example Usage

```bash
# Train a dynamic model with multiple task types and custom parameters
python -m grpo.dynamic_qwen0 --learning_rate 2e-5 --lora_r 32 --lora_alpha 64 --dropout 0.05 --wandb_project "math_dynamic_training"

# Train a model specifically for programming tasks with larger batch size
python -m grpo.programming_qwen0 --batch_size 8 --gradient_accumulation_steps 4 --max_seq_length 2048 --save_steps 200

# Train a model for tutoring and error identification with custom dataset
python -m grpo.tutor_grpo --dataset "username/math_errors_dataset" --epochs 5 --eval_steps 100 --max_steps 10000
```

## Model Export and Deployment

After training, models can be exported using `auxilary.export_model.py` with various options:
```bash
# Basic export with default settings
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/dynamic_0_20240315 --output-dir models/dynamic_0

# Export with quantization for deployment efficiency
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/tutor_20240315 --output-dir models/tutor --quantize --bits 4

# Export with specific adapter configuration
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/programming_20240315 --output-dir models/programming --adapter-name "programming_adapter"
```

Exported models can then be evaluated using the benchmark scripts to measure improvement:
```bash
# Evaluate exported model on standard benchmark
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000 --dataset Metaskepsis/validation_set --produce-statistics

# Compare performance against baseline models
python -m benchmarks.programming_benchmark --main LOCAL_0 --main-port 8000 --auxiliary LOCAL_1 --auxiliary-port 8001 --best-of 3
```

## Advanced Features

### Curriculum Learning
Dynamic training scripts support curriculum learning with increasing difficulty:
```bash
python -m grpo.dynamic_qwen0 --curriculum --curriculum_steps 1000,2000,3000 --curriculum_difficulties easy,medium,hard
```

### Distributed Training
For multi-GPU setups, training scripts support distributed training:
```bash
python -m torch.distributed.launch --nproc_per_node=4 grpo.dynamic_qwen0 --distributed_training
```

### Custom Reward Components
Reward weights can be customized for specific training objectives:
```bash
python -m grpo.solution_qwen0 --correctness_weight 2.0 --clarity_weight 1.0 --step_weight 1.5
```

### Checkpoint Merging
Merge multiple specialized models into a single model:
```bash
python -m auxilary.merge_models --models models/solution,models/programming,models/tutor --output models/combined --weights 0.4,0.3,0.3
```
