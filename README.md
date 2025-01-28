# Numina-Olympiads Benchmark Suite

A comprehensive benchmark suite for evaluating mathematical problem-solving models, particularly focused on olympiad-style problems. The suite provides tools for generating solutions, validating answers, and running benchmarks to assess model performance.

## Overview

This repository contains a set of tools and scripts for generating and evaluating mathematical solutions to olympiad-style problems. The suite includes:

1. **Adversarial Solution Generation**: Tools to generate both correct and deliberately incorrect but convincing solutions
2. **Automated Verification**: Scripts to validate solutions using numeric and structural checks
3. **Benchmarking Framework**: Comprehensive benchmarking tools to evaluate model performance
4. **Dataset Utilities**: Tools for processing and preparing datasets of mathematical problems

## Features

- **Adversarial Generators**:
  - `AlternatingGenerator`: Generates solution pairs by alternating between correct and adversarial solutions
  - `AdversarialGenerator`: Creates pairs of valid correct and incorrect solutions

- **Benchmarking Tools**:
  - `benchmark.py`: Main benchmarking script for evaluating model performance
  - `tournament_benchmark.py`: Implements tournament-style evaluation of solutions

- **Dataset Utilities**:
  - `process_dataset.py`: Processes and filters datasets to ensure high-quality examples
  - `prepare_for_SFT.py`: Prepares datasets for Self-Supervised Fine-Tuning (SFT)
  - `shuffle_dataset.py`: Shuffles and reassigns IDs to dataset examples

- **Helper Scripts**:
  - `count_entries.py`: Counts and analyzes entries in dataset files
  - `filtering.py`: Filters dataset entries based on various criteria
  - `merge_json.py`: Merges multiple JSON files into a single dataset

## Requirements

- Python 3.8+
- asyncio
- typing
- dotenv
- langchain
- sympy
- latex2sympy2
- aiohttp
- tqdm
- datasets
- numpy

## Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

The benchmark suite uses environment variables for configuration. Create a `.env` file with the following settings:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 3. Running Benchmarks

To run the main benchmark:

```bash
python benchmark.py
```

To run the tournament benchmark:

```bash
python tournament_benchmark.py
```

### 4. Generating Solutions

Use the adversarial generator to create solution pairs:

```python
from adversarial_generator import AdversarialGenerator

generator = AdversarialGenerator()
solutions = await generator.generate(problem="Your math problem here", correct_answer="Final answer")
```

### 5. Processing Datasets

Prepare datasets for training:

```python
from process_dataset import process_dataset

process_dataset(input_path="input.json", output_path="output.json")
```

## Dataset

The suite uses the Numina-Olympiads dataset, which is a filtered version of the NuminaMath-CoT dataset containing only olympiad problems with valid answers. The dataset includes:

- Train split: 21,408 examples
- All examples contain valid boxed answers
- Problems include detailed solutions with step-by-step reasoning

## Configuration Options

The benchmark suite provides various configuration options through command-line arguments:

```bash
python benchmark.py --help
```

Key options include:
- `--main`: Main model to use for solving problems
- `--auxiliary`: Auxiliary model for judging solutions
- `--max-concurrent`: Maximum number of concurrent problems
- `--best-of`: Number of attempts per problem
- `--tolerance`: Tolerance for numeric answer comparison

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.
