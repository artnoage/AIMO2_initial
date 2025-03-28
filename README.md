# Mathematical Problem-Solving Framework

A comprehensive framework for evaluating and improving mathematical problem-solving capabilities of language models through benchmarking, reinforcement learning, and dataset processing.

## Project Structure

The project is organized into three main components:

### 1. [Benchmarks](/benchmarks)
Evaluation scripts for different mathematical problem-solving capabilities:
- **Standard Benchmark**: Step-by-step mathematical solutions with answer verification
- **Programming Benchmark**: Python code generation for mathematical problems
- **Test Benchmark**: Test function creation for solution verification
- **Architect Benchmark**: Architectural analysis and planning for complex problems
- **Tutor Benchmark**: Error identification and correction in mathematical solutions
- **Step Benchmark**: Step-by-step solution analysis to identify reasoning errors

### 2. [GRPO Training](/grpo)
Generative Reinforcement Policy Optimization scripts for training models:
- **Dynamic Training**: Multi-task training with dynamic reward selection
- **Task-Specific Training**: Specialized training for solutions, programming, testing, tutoring
- **Reward Functions**: Customized rewards for different mathematical tasks
- **Statistics Tracking**: Detailed monitoring of training progress and reward distributions

### 3. [Utilities](/utils) and [Auxiliary Tools](/auxilary)
Support modules and data processing tools:
- **Agent Implementations**: Specialized agents for different mathematical tasks
- **Model Utilities**: Interfaces for local and API-based language models
- **Solution Validation**: Mathematical answer verification and step analysis
- **Dataset Processing**: Filtering, merging, and preparation of training data
- **Progress Tracking**: Monitoring and reporting of benchmark performance

## Key Features

### Comprehensive Benchmarking
- Evaluate mathematical problem-solving with step-by-step verification
- Test code generation for mathematical problems
- Assess error identification and correction abilities
- Compare multiple solution approaches
- Track detailed performance metrics across different model types

### Advanced Training Framework
- Multi-task reinforcement learning with dynamic rewards
- Parameter-efficient fine-tuning with LoRA adapters
- Integration with Unsloth for efficient training
- Wandb logging for experiment tracking
- Diverse reward components for solution quality, correctness, and style

### Extensive Data Processing
- Dataset filtering and validation for high-quality training
- Answer extraction and verification from LaTeX expressions
- Multiple-choice problem detection and handling
- Solution similarity analysis for diversity measurement
- HuggingFace dataset integration for easy sharing and loading

## Usage Examples

### Running Benchmarks

```bash
# Standard mathematical solutions benchmark
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000 --dataset Metaskepsis/Numina

# Programming solutions benchmark
python -m benchmarks.programming_benchmark --main GPT --auxiliary CLAUDE --best-of 3

# Tutor benchmark for error identification
python -m benchmarks.tutor_benchmark --main LOCAL_0 --main-port 8000 --max-concurrent 32
```

### Training Models with GRPO

```bash
# Multi-task training with dynamic rewards
python -m grpo.dynamic_qwen0

# Programming-specific training
python -m grpo.programming_qwen0

# Tutor training for error identification
python -m grpo.tutor_grpo
```

### Processing Datasets

```bash
# Filter a dataset based on criteria
python -m auxilary.filter_dataset --repo-name Metaskepsis/Olympiads_hard --output-dir olympiads_filtered

# Merge multiple JSON files
python -m auxilary.merge_json results_folder --output merged.json

# Create a validation dataset
python -m auxilary.create_validation_set
```

## Configuration

### Model Configuration
Models can be configured through command-line arguments:
- Local models via port specification (`--main-port`, `--auxiliary-port`)
- Cloud models through API keys (set via environment variables)
- Model temperatures and other parameters for generation quality

### Benchmark Configuration
Benchmarks support various configuration options:
- Dataset selection and filtering
- Concurrency and timeout settings
- Answer tolerance for numeric verification
- Output formats and statistics generation

### Training Configuration
Training scripts configure:
- Learning rates and optimization parameters
- Batch sizes and sequence lengths
- Reward components and weights
- Checkpoint frequency and model saving

## Requirements

- Python 3.8+
- PyTorch
- Transformers and Unsloth for training
- Datasets library for HuggingFace integration
- Sympy for mathematical expression evaluation
- OpenRouter API key (for cloud model access)

## Documentation

Each directory contains its own README with detailed information:
- [Benchmarks README](/benchmarks/README.md): Details on benchmark types and configuration
- [GRPO README](/grpo/README.md): Information on training scripts and reward functions
- [Utils README](/utils/README.md): Documentation for utility modules
- [Auxiliary README](/auxilary/README.md): Guide to dataset processing and model management

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## License

MIT License - See LICENSE file for details
