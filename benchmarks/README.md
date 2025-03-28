# Mathematical Problem Solving Benchmarks

This directory contains various benchmark scripts for evaluating mathematical problem-solving capabilities of language models.

## Benchmark Types

### Standard Benchmark (`standard_benchmark.py`)
Evaluates a model's ability to solve mathematical problems and provide correct answers. It generates multiple solutions for each problem and verifies the answers numerically.

### Programming Benchmark (`programming_benchmark.py`)
Tests a model's ability to write Python code that solves mathematical problems. The code is executed to verify correctness.

### Test Benchmark (`test_benchmark.py`)
Evaluates a model's ability to create test functions that can verify mathematical solutions. These test functions should correctly identify valid and invalid answers.

### Architect Benchmark (`architect_benchmark.py`)
Tests a pipeline approach where one model acts as an "architect" to analyze problems and create guidance, and another model implements the solution as code.

### Tutor Benchmark (`tutor_benchmark.py`)
Evaluates a model's ability to identify errors in mathematical solutions and provide corrections. It simulates a tutoring scenario where incorrect solutions are analyzed.

### Step Benchmark (`step_benchmark.py`)
Analyzes solutions step-by-step to identify the first incorrect step in a solution. This helps understand where reasoning errors occur in the solution process.

## Common Features

All benchmarks share these common features:
- Support for multiple model configurations via command-line arguments
- Parallel processing of examples
- Detailed logging and statistics
- Result saving in various formats
- Progress tracking during long benchmark runs

## Usage

Run any benchmark with the `--help` flag to see available options:

```bash
python -m benchmarks.standard_benchmark --help
```

Example usage:

```bash
python -m benchmarks.programming_benchmark --model OPENAI_GPT4 --dataset Metaskepsis/validation_set_filtered --best_of 3
```

## Configuration

Benchmarks use the `BenchmarkConfig` class from `utils.benchmark_config` for configuration. Key parameters include:
- `model`: The model to use (e.g., OPENAI_GPT4, ANTHROPIC_CLAUDE, etc.)
- `dataset`: HuggingFace dataset to use for benchmarking
- `best_of`: Number of solutions to generate per problem
- `max_examples`: Maximum number of examples to process
- `max_concurrent`: Maximum number of concurrent requests
- `timeout`: Timeout for code execution in seconds
- `tolerance`: Numeric tolerance for answer verification
