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

- `DualProofReward`: Rewards for dual proof solutions (logical proof + code)
  - Evaluates correctness of both logical proof and code implementation
  - Provides additional bonus for having both components correct
  - Assesses structure and organization of the dual solution

- `TestDrivenProgrammerReward`: Rewards for test-driven programming solutions
  - Evaluates test suite quality and implementation correctness
  - Assesses test coverage and implementation alignment
  - Rewards comprehensive test-driven development approach

Each reward function corresponds to a specific benchmark in the `benchmarks` directory and uses the same evaluation criteria to ensure consistency between training and evaluation.

### Reflective Solver Reward (`solver_ref_reward.py`)
A specialized reward function for reflective problem-solving:
- Evaluates both solution correctness and self-assessment accuracy
- Rewards correct solutions with accurate self-assessment
- Penalizes incorrect solutions with overconfident self-assessment
- Tracks detailed reflection statistics including true/false positives/negatives
- Integrates with Wandb for visualization of reflection metrics

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

### Task-Specific Training
- `programming_qwen0.py`: Train for Python code generation
  - Corresponds to `benchmarks.programming_benchmark`
  - Uses `utils.data_preparation.prepare_programming_data`
  - Includes code execution verification in the training loop
  - Rewards efficient and readable code implementations
  - Integrates with Wandb for tracking training metrics
  - Uses Unsloth for efficient training of Qwen models
  - Implements LoRA fine-tuning for parameter-efficient training

### Reference Solvers
- `solver_ref_4b.py`, `solver_ref_8b.py`, `solver_ref_14b.py`: Reference solver implementations
  - Provide baseline models for comparison
  - Used for generating reference solutions
  - Support different model sizes (4B, 8B, 14B)

- `solver_ver_7B.py`: Verification solver implementation
  - Used for verifying solutions
  - Implements 7B parameter model

- `solver_ref_reward.py`: Reflective solver reward implementation
  - Evaluates solution correctness and self-assessment
  - Tracks detailed reflection statistics

## Integration with Other Components

### Connection to Benchmarks
Each training script is designed to improve performance on specific benchmarks:
- `programming_qwen0.py` → Improves performance on `benchmarks.programming_benchmark.py`

### Utility Dependencies
Training scripts rely on utility modules in the `utils` directory:
- `utils.data_preparation`: Formats datasets for different training tasks with task-specific processing
- `utils.solution_utils`: Validates solutions and extracts answers with LaTeX support
- `utils.similarity_checker`: Measures diversity between solutions using embedding models
- `utils.model_utils`: Handles model responses and timeouts with retry logic
- `utils.agents`: Provides system prompts and agent implementations

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
python -m grpo.programming_qwen0 --learning_rate 2e-5 --epochs 3 --batch_size 4
```

## Configuration

Training scripts use the `RewardConfig` class from `config.py` for configuration. Key parameters include:

### Model Settings
- `model_type`: Type of model to train (e.g., "programming")
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
# Train a model specifically for programming tasks with larger batch size
python -m grpo.programming_qwen0 --batch_size 8 --gradient_accumulation_steps 4 --max_seq_length 2048 --save_steps 200
```

## Model Export and Deployment

After training, models can be exported using `auxilary.export_model.py` with various options:
```bash
# Basic export with default settings
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/programming_20240315 --output-dir models/programming

# Export with quantization for deployment efficiency
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/programming_20240315 --output-dir models/programming --quantize --bits 4
```

Exported models can then be evaluated using the benchmark scripts to measure improvement:
```bash
# Evaluate exported model on programming benchmark
python -m benchmarks.programming_benchmark --main LOCAL_0 --main-port 8000 --dataset Metaskepsis/validation_set --produce-statistics
