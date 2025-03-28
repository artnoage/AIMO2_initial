# Mathematical Problem Solving Benchmarks

This directory contains various benchmark scripts for evaluating mathematical problem-solving capabilities of language models. These benchmarks form the evaluation component of the larger mathematical problem-solving framework.

## Benchmark Types

### Standard Benchmark (`standard_benchmark.py`)
Evaluates a model's ability to solve mathematical problems and provide correct answers. It generates multiple solutions for each problem and verifies the answers numerically.
- Uses `utils.solution_utils.extract_numeric_answer` for answer verification
- Supports LaTeX notation in answers
- Generates detailed solution statistics

### Programming Benchmark (`programming_benchmark.py`)
Tests a model's ability to write Python code that solves mathematical problems. The code is executed to verify correctness.
- Uses `utils.solution_utils.run_code_safely` for secure code execution
- Handles timeouts and execution errors
- Verifies numeric answers against expected results

### Test Benchmark (`test_benchmark.py`)
Evaluates a model's ability to create test functions that can verify mathematical solutions. These test functions should correctly identify valid and invalid answers.
- Uses `utils.solution_utils.run_test_function` to evaluate test functions
- Generates test cases automatically
- Measures both correctness and robustness of test functions

### Architect Benchmark (`architect_benchmark.py`)
Tests a pipeline approach where one model acts as an "architect" to analyze problems and create guidance, and another model implements the solution as code.
- Uses two models in sequence (architect and programmer)
- Measures the effectiveness of problem decomposition
- Evaluates the quality of generated code

### Tutor Benchmark (`tutor_benchmark.py`)
Evaluates a model's ability to identify errors in mathematical solutions and provide corrections. It simulates a tutoring scenario where incorrect solutions are analyzed.
- Identifies errors in mathematical reasoning
- Provides explanations and corrections
- Measures accuracy of error detection

### Step Benchmark (`step_benchmark.py`)
Analyzes solutions step-by-step to identify the first incorrect step in a solution. This helps understand where reasoning errors occur in the solution process.
- Breaks solutions into individual steps
- Identifies the exact point of reasoning failure
- Provides insights for targeted improvement

## Integration with Other Components

### Connection to GRPO Training
The benchmarks provide evaluation metrics that inform the reward functions used in the GRPO training scripts:
- `standard_benchmark.py` → `grpo.solution_qwen0.py`
- `programming_benchmark.py` → `grpo.programming_qwen0.py`
- `test_benchmark.py` → `grpo.test_programming_qwen0.py`
- `tutor_benchmark.py` → `grpo.tutor_grpo.py`

### Utility Dependencies
Benchmarks rely heavily on the utility modules in the `utils` directory:
- `utils.agents`: Agent implementations for different tasks
- `utils.benchmark_config`: Configuration parsing and management
- `utils.model_utils`: Model interfaces and response handling
- `utils.progress_tracker`: Result tracking and statistics
- `utils.solution_utils`: Solution validation and verification

### Dataset Processing
Benchmarks use datasets processed by the auxiliary scripts:
- Filtered datasets from `auxilary.filter_dataset`
- Validation sets from `auxilary.create_validation_set`
- Merged datasets from `auxilary.merge_json`

## Common Features

All benchmarks share these common features:
- Support for multiple model configurations via command-line arguments
- Parallel processing of examples using asyncio
- Detailed logging and statistics generation
- Result saving in various formats (JSON, CSV, HuggingFace datasets)
- Progress tracking during long benchmark runs
- Timeout handling for model responses and code execution

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
# Standard benchmark with a local model
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000 --main-temp 0.9 --dataset Metaskepsis/Numina --split validation

# Programming benchmark with cloud models
python -m benchmarks.programming_benchmark --main GPT --auxiliary CLAUDE --dataset Metaskepsis/Numina --best-of 3 --max-concurrent 32

# Tutor benchmark with result dataset creation
python -m benchmarks.tutor_benchmark --main LOCAL_0 --main-port 8000 --create-dataset --produce-statistics
```

## Output and Analysis

Benchmark results are saved in the following formats:
- JSON files with detailed results for each problem
- Statistics summary in CSV format
- HuggingFace datasets for further analysis or training
- Console output with progress and summary statistics

These results can be used to:
1. Evaluate model performance on mathematical reasoning
2. Generate training data for GRPO fine-tuning
3. Identify specific areas for model improvement
4. Compare different model architectures and configurations
