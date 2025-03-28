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

Run any benchmark script directly:

```bash
python -m benchmarks.standard_benchmark
```

## Configuration

Benchmarks use the `BenchmarkConfig` class from `utils.benchmark_config` for configuration. Key parameters include:

### Model Selection
- `--main`: Main model to use (e.g., LOCAL_0, CLAUDE, GPT)
- `--auxiliary`: Auxiliary model for judging or secondary tasks
- `--auxiliary2`: Second auxiliary model (optional)
- `--main-temp`: Temperature for main model (default: 0.7)
- `--auxiliary-temp`: Temperature for auxiliary model (default: 0.7)
- `--main-port`: Port for main model server (for local models)
- `--auxiliary-port`: Port for auxiliary model server

### Dataset Options
- `--dataset`: HuggingFace dataset to use (default: Metaskepsis/Numina)
- `--split`: Dataset split to use (train/validation/test)
- `--source`: Filter problems by source
- `--seed`: Seed for dataset operations

### Execution Settings
- `--max-concurrent`: Maximum number of concurrent problems (default: 64)
- `--best-of`: Number of attempts per problem (default: 1)
- `--completions`: Number of completions to try per path (default: 20)
- `--timeout`: Timeout in seconds for code execution (default: 10)
- `--tolerance`: Tolerance for numeric answer comparison (default: 1e-2)

### Output Settings
- `--produce-statistics`: Generate detailed statistics file
- `--create-dataset`: Create a HuggingFace dataset from results
- `--upload-dataset`: Upload the created dataset to HuggingFace Hub

## Example Usage

```bash
python -m benchmarks.programming_benchmark --main GPT --auxiliary CLAUDE --dataset Metaskepsis/Numina --best-of 3 --max-concurrent 32
```

```bash
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000 --main-temp 0.9 --dataset Metaskepsis/Numina --split validation
```
