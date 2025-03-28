# Mathematical Problem-Solving Framework

A comprehensive framework for evaluating and improving mathematical problem-solving capabilities of language models through benchmarking, reinforcement learning, and dataset processing.

## Project Structure

The project is organized into three main components:

### 1. [Benchmarks](/benchmarks)
Evaluation scripts for different mathematical problem-solving capabilities:
- **Standard Benchmark**: Step-by-step mathematical solutions with answer verification and numeric validation
- **Programming Benchmark**: Python code generation for mathematical problems with secure execution
- **Test Benchmark**: Test function creation for solution verification with automatic test case generation
- **Architect Benchmark**: Architectural analysis and planning for complex problems using a two-stage approach
- **Tutor Benchmark**: Error identification and correction in mathematical solutions with detailed feedback
- **Step Benchmark**: Step-by-step solution analysis to identify reasoning errors at specific solution stages

### 2. [GRPO Training](/grpo)
Generative Reinforcement Policy Optimization scripts for training models:
- **Dynamic Training**: Multi-task training with dynamic reward selection based on example type
- **Task-Specific Training**: Specialized training for solutions, programming, testing, tutoring with focused rewards
- **Reward Functions**: Customized rewards for different mathematical tasks with component-based scoring
- **Statistics Tracking**: Detailed monitoring of training progress and reward distributions with Wandb integration

### 3. [Utilities](/utils) and [Auxiliary Tools](/auxilary)
Support modules and data processing tools:
- **Agent Implementations**: Specialized agents for different mathematical tasks with consistent interfaces
- **Model Utilities**: Interfaces for local and API-based language models with timeout and retry handling
- **Solution Validation**: Mathematical answer verification and step analysis with LaTeX support
- **Dataset Processing**: Filtering, merging, and preparation of training data with quality controls
- **Progress Tracking**: Monitoring and reporting of benchmark performance with real-time updates

## Key Features

### Comprehensive Benchmarking
- Evaluate mathematical problem-solving with step-by-step verification and numeric answer validation
- Test code generation for mathematical problems with secure execution environment
- Assess error identification and correction abilities in incorrect solutions
- Compare multiple solution approaches with diversity metrics
- Track detailed performance metrics across different model types and configurations
- Support for both local and cloud-based language models

### Advanced Training Framework
- Multi-task reinforcement learning with dynamic rewards based on example type
- Parameter-efficient fine-tuning with LoRA adapters for memory efficiency
- Integration with Unsloth for 2-3x faster training of Qwen models
- Wandb logging for experiment tracking with detailed reward component visualization
- Diverse reward components for solution quality, correctness, style, and diversity
- Checkpoint management with automatic merging and export

### Extensive Data Processing
- Dataset filtering and validation for high-quality training with configurable criteria
- Answer extraction and verification from LaTeX expressions with sympy integration
- Multiple-choice problem detection and handling with specialized processing
- Solution similarity analysis for diversity measurement using embedding models
- HuggingFace dataset integration for easy sharing, loading, and version control
- Robust error handling and validation for dataset processing

## Usage Examples

### Running Benchmarks

```bash
# Standard mathematical solutions benchmark
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000 --dataset Metaskepsis/Numina --max-concurrent 64 --best-of 3

# Programming solutions benchmark with cloud models
python -m benchmarks.programming_benchmark --main GPT --auxiliary CLAUDE --best-of 3 --timeout 30 --produce-statistics

# Tutor benchmark for error identification with dataset creation
python -m benchmarks.tutor_benchmark --main LOCAL_0 --main-port 8000 --max-concurrent 32 --create-dataset --split validation
```

### Training Models with GRPO

```bash
# Multi-task training with dynamic rewards
python -m grpo.dynamic_qwen0 --learning_rate 2e-5 --epochs 3 --batch_size 4

# Programming-specific training with custom configuration
python -m grpo.programming_qwen0 --model_name unsloth/Qwen1.5-7B --lora_r 32 --gradient_accumulation_steps 8

# Tutor training for error identification with wandb logging
python -m grpo.tutor_grpo --wandb_project "math_tutor_training" --save_steps 500 --eval_steps 100
```

### Processing Datasets

```bash
# Filter a dataset based on criteria with hub upload
python -m auxilary.filter_dataset --repo-name Metaskepsis/Olympiads_hard --output-dir olympiads_filtered --exclude-multiple-choice --max-length 2000 --push-to-hub-name "username/filtered_olympiads"

# Merge multiple JSON files with custom output
python -m auxilary.merge_json results_folder --output merged.json --pretty-print

# Create a validation dataset with specific sources
python -m auxilary.create_validation_set --sources Metaskepsis/Numina,Metaskepsis/Olympiads --output-repo "username/validation_set" --sample-size 500
```

## Configuration

### Model Configuration
Models can be configured through command-line arguments:
- Local models via port specification (`--main-port`, `--auxiliary-port`) for self-hosted instances
- Cloud models through API keys (set via environment variables: `OPENROUTER_API_KEY`, `HUGGINGFACE_TOKEN`)
- Model temperatures and other parameters for generation quality and diversity control
- Support for multiple model types: LOCAL_0, LOCAL_1, GPT, CLAUDE, GEMINI, MISTRAL, etc.

### Benchmark Configuration
Benchmarks support various configuration options:
- Dataset selection and filtering with source-specific options
- Concurrency and timeout settings for parallel processing
- Answer tolerance for numeric verification with configurable precision
- Output formats and statistics generation with detailed metrics
- Result saving in multiple formats (JSON, CSV, HuggingFace datasets)

### Training Configuration
Training scripts configure:
- Learning rates and optimization parameters (AdamW with weight decay)
- Batch sizes and gradient accumulation steps for memory efficiency
- Reward components and weights for different aspects of solutions
- Checkpoint frequency and model saving with automatic merging
- LoRA parameters (rank, alpha, dropout) for parameter-efficient training

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers and Unsloth for training
- Datasets library for HuggingFace integration
- Sympy for mathematical expression evaluation
- NLTK for text processing
- Sentence-Transformers for embedding generation
- Wandb for experiment tracking
- OpenRouter API key (for cloud model access)
- HuggingFace token (for dataset and model uploads)

## Documentation

Each directory contains its own README with detailed information:
- [Benchmarks README](/benchmarks/README.md): Details on benchmark types, configuration, and evaluation metrics
- [GRPO README](/grpo/README.md): Information on training scripts, reward functions, and hyperparameters
- [Utils README](/utils/README.md): Documentation for utility modules, model interfaces, and solution validation
- [Auxiliary README](/auxilary/README.md): Guide to dataset processing, model management, and data transformation

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Before submitting, please ensure:
- Code follows project style guidelines
- All tests pass
- Documentation is updated
- Changes are backward compatible when possible

## License

MIT License - See LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```
@software{mathematical_problem_solving_framework,
  author = {Metaskepsis Team},
  title = {Mathematical Problem-Solving Framework},
  year = {2025},
  url = {https://github.com/Metaskepsis/math-problem-solving}
}
```
