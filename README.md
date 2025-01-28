# Mathematical Problem-Solving Benchmark Suite & Training Data Framework

A comprehensive framework for evaluating mathematical problem-solving capabilities of language models and generating high-quality training data for reinforcement learning (RL).

## Overview

This suite provides tools for:

1. **Solution Generation & Verification**: Generate and validate mathematical solutions with step-by-step reasoning
2. **Tournament Evaluation**: Compare solutions through tournament-style competitions
3. **Progress Tracking**: Monitor and analyze benchmark performance
4. **Dataset Processing**: Tools for filtering and preparing mathematical problem datasets
5. **RL Training Data Generation**: Create training examples for improving model reasoning and problem-solving

## Key Components

### Benchmarking Tools
- **benchmark.py**: Main benchmarking script for evaluating model performance
- **tournament_benchmark.py**: Tournament-style evaluation of solutions

### Dataset Utilities
- **process_dataset.py**: Processes datasets to ensure high-quality examples with valid answers
- **filtering.py**: Filters dataset entries based on various criteria
- **merge_json.py**: Merges multiple JSON files into a single dataset
- **shuffle_dataset.py**: Shuffles and reassigns IDs to dataset examples

### Training Data Generation
- **adversarial_generator.py**: Generates pairs of correct and incorrect solutions using multiple agents
- **alternating_generator.py**: Alternates between solver and adversarial agents to create training examples

### Agents
- **Analysis Agent**: Provides problem analysis and approach strategies
- **Step Agent**: Generates individual solution steps
- **Completion Agent**: Completes partial solutions
- **Judge Agent**: Evaluates and compares solution quality
- **Loki Agent**: Generates deliberately incorrect but convincing solutions

### Utilities
- **Numeric Verification**: Validates mathematical answers with configurable tolerance
- **Step Analysis**: Validates solution structure and step coherence
- **Progress Tracking**: Real-time statistics and performance monitoring
- **Tournament Management**: Organizes solution competitions with judging

## Requirements

- Python 3.8+
- Key packages: asyncio, sympy, latex2sympy2, aiohttp, tqdm, datasets
- OpenRouter API key (for cloud model access)

## Configuration

The suite supports both local and cloud-based models through a flexible configuration system:

```python
config = BenchmarkConfig(
    main="LOCAL",              # Main solving model
    auxiliary="LOCAL_2",       # Auxiliary/judging model
    main_port=8000,           # Local model ports
    auxiliary_port=6000,
    max_concurrent=256,       # Concurrent processing
    best_of=40,              # Solutions per problem
    completions=35,          # Completion attempts
    tolerance=1e-6           # Answer comparison tolerance
)
```

### Model Options
- Local deployments (ports 8000/6000)
- OpenRouter API models (requires API key)
- Multiple model types (Claude, GPT, Gemini, etc.)

## Usage

### 1. Environment Setup

```bash
export OPENROUTER_API_KEY=your_key_here  # If using cloud models
```

### 2. Running Benchmarks

Standard benchmark:
```bash
python benchmark.py --main LOCAL --auxiliary LOCAL_2 --max-concurrent 256
```

Tournament evaluation:
```bash
python tournament_benchmark.py --main LOCAL --auxiliary LOCAL_2 --best-of 40
```

### 3. Dataset Processing

Process and filter dataset:
```bash
python auxilary/process_dataset.py --dataset input_dataset --output-dir processed
```

Filter entries:
```bash
python auxilary/filtering.py input.json output.json --types light dark --success-rate-above 0.8
```

Merge multiple files:
```bash
python auxilary/merge_json.py results_folder --output merged.json
```

Shuffle dataset:
```bash
python auxilary/shuffle_dataset.py input.json output.json --seed 42
```

## Features

### Solution Validation
- Step-by-step structure verification
- LaTeX mathematical notation support
- Numeric answer comparison with tolerance
- Multiple-choice problem detection

### Progress Tracking
- Real-time statistics
- Success rate monitoring
- Judge accuracy tracking
- Tournament performance analysis

### Dataset Support
- HuggingFace dataset integration
- Local dataset caching
- Filtered problem selection
- Progress persistence

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## License

MIT License - See LICENSE file for details
