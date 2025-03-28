# Mathematical Problem Solving Benchmarks

This directory contains various benchmark scripts for evaluating mathematical problem-solving capabilities of language models. These benchmarks form the evaluation component of the larger mathematical problem-solving framework.

## Benchmark Types

### Standard Benchmark (`standard_benchmark.py`)
Evaluates a model's ability to solve mathematical problems and provide correct answers. It generates multiple solutions for each problem and verifies the answers numerically.
- Uses `utils.solution_utils.extract_numeric_answer` for answer verification with sympy integration
- Supports LaTeX notation in answers with comprehensive parsing
- Generates detailed solution statistics with step-by-step analysis
- Measures solution quality, correctness, and explanation clarity
- Handles multiple answer formats (boxed, hash-marked, inline)
- Supports group-based evaluation for solution diversity measurement
- Implements majority voting for answer verification with configurable thresholds

### Programming Benchmark (`programming_benchmark.py`)
Tests a model's ability to write Python code that solves mathematical problems. The code is executed to verify correctness.
- Uses `utils.solution_utils.run_code_safely` for secure code execution in isolated environments
- Handles timeouts and execution errors with detailed error reporting
- Verifies numeric answers against expected results with configurable tolerance
- Evaluates code quality metrics (complexity, readability, efficiency)
- Supports multiple solution approaches with comparative analysis
- Implements memory and CPU usage limits for secure execution
- Provides detailed execution traces for debugging and analysis

### Test Benchmark (`test_benchmark.py`)
Evaluates a model's ability to create test functions that can verify mathematical solutions. These test functions should correctly identify valid and invalid answers.
- Uses `utils.solution_utils.run_test_function` to evaluate test functions on multiple test cases
- Generates test cases automatically with edge case detection
- Measures both correctness and robustness of test functions
- Evaluates test coverage and edge case handling
- Supports both numeric and symbolic test verification
- Implements timeout protection for infinite loops
- Provides detailed analysis of test function behavior

### Architect Benchmark (`architect_benchmark.py`)
Tests a pipeline approach where one model acts as an "architect" to analyze problems and create guidance, and another model implements the solution as code.
- Uses two models in sequence (architect and programmer) with configurable interaction
- Measures the effectiveness of problem decomposition and planning
- Evaluates the quality of generated code and implementation fidelity
- Supports iterative refinement between architect and programmer
- Analyzes the impact of architectural guidance on solution quality
- Implements comparative analysis with single-model approaches
- Provides detailed interaction logs for process analysis

### Tutor Benchmark (`tutor_benchmark.py`)
Evaluates a model's ability to identify errors in mathematical solutions and provide corrections. It simulates a tutoring scenario where incorrect solutions are analyzed.
- Identifies errors in mathematical reasoning with precision
- Provides explanations and corrections with pedagogical quality assessment
- Measures accuracy of error detection and correction effectiveness
- Evaluates explanation clarity and helpfulness
- Supports multiple error types (conceptual, computational, logical)
- Implements scoring for constructive feedback quality
- Provides comparative analysis with expert-identified errors

### Step Benchmark (`step_benchmark.py`)
Analyzes solutions step-by-step to identify the first incorrect step in a solution. This helps understand where reasoning errors occur in the solution process.
- Breaks solutions into individual steps with logical boundary detection
- Identifies the exact point of reasoning failure with detailed analysis
- Provides insights for targeted improvement of reasoning capabilities
- Measures step-by-step correctness with cumulative evaluation
- Supports partial credit for partially correct solutions
- Implements visualization of reasoning paths and failure points
- Provides aggregated statistics on common failure patterns

## Integration with Other Components

### Connection to GRPO Training
The benchmarks provide evaluation metrics that inform the reward functions used in the GRPO training scripts:
- `standard_benchmark.py` → `grpo.solution_qwen0.py` (step-by-step solution generation)
- `programming_benchmark.py` → `grpo.programming_qwen0.py` (code generation for math problems)
- `test_benchmark.py` → `grpo.test_programming_qwen0.py` (test function creation)
- `tutor_benchmark.py` → `grpo.tutor_grpo.py` (error identification and correction)
- `architect_benchmark.py` → `grpo.dynamic_qwen0.py` (architectural planning component)
- `step_benchmark.py` → Used for analysis across all training types

The benchmarks generate detailed performance metrics that can be used to:
- Identify specific weaknesses in model capabilities
- Measure improvement from GRPO training iterations
- Generate new training examples from failure cases
- Compare different training approaches and configurations
- Provide targeted feedback for reward function refinement

### Utility Dependencies
Benchmarks rely heavily on the utility modules in the `utils` directory:
- `utils.agents`: Agent implementations for different tasks with consistent interfaces
- `utils.benchmark_config`: Configuration parsing and management with extensive options
- `utils.model_utils`: Model interfaces and response handling with timeout protection
- `utils.progress_tracker`: Result tracking and statistics with real-time updates
- `utils.solution_utils`: Solution validation and verification with comprehensive checks
- `utils.logger`: Structured logging with configurable verbosity
- `utils.similarity_checker`: Solution diversity measurement for group-based evaluation

### Dataset Processing
Benchmarks use datasets processed by the auxiliary scripts:
- Filtered datasets from `auxilary.filter_dataset` with quality controls
- Validation sets from `auxilary.create_validation_set` for consistent evaluation
- Merged datasets from `auxilary.merge_json` for comprehensive testing
- Converted datasets from `auxilary.datatype_transformation` for format compatibility
- Custom datasets created from previous benchmark runs for targeted testing

## Common Features

All benchmarks share these common features:
- Support for multiple model configurations via command-line arguments with extensive options
- Parallel processing of examples using asyncio with configurable concurrency
- Detailed logging and statistics generation with component breakdowns
- Result saving in various formats (JSON, CSV, HuggingFace datasets) with configurable paths
- Progress tracking during long benchmark runs with ETA estimation
- Timeout handling for model responses and code execution with graceful recovery
- Caching mechanisms for efficient re-runs and incremental testing
- Comprehensive error handling with detailed diagnostics
- Support for both local and cloud-based language models
- Configurable verbosity levels for different use cases

## Usage

Run any benchmark script directly with optional command-line arguments:

```bash
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000
```

## Configuration

Benchmarks use the `BenchmarkConfig` class from `utils.benchmark_config` for configuration. Key parameters include:

### Model Selection
- `--main`: Main model to use (e.g., LOCAL_0, CLAUDE, GPT, GEMINI, MISTRAL)
- `--auxiliary`: Auxiliary model for judging or secondary tasks
- `--auxiliary2`: Second auxiliary model for comparative analysis or verification
- `--main-temp`: Temperature for main model (default: 0.7) for controlling randomness
- `--auxiliary-temp`: Temperature for auxiliary model (default: 0.7)
- `--main-port`: Port for main model server (for local models)
- `--auxiliary-port`: Port for auxiliary model server
- `--main-max-tokens`: Maximum tokens for main model responses
- `--auxiliary-max-tokens`: Maximum tokens for auxiliary model responses

### Dataset Options
- `--dataset`: HuggingFace dataset to use (default: Metaskepsis/Numina)
- `--split`: Dataset split to use (train/validation/test)
- `--source`: Filter problems by source (e.g., AIME, IMO, Putnam)
- `--seed`: Seed for dataset operations and random sampling
- `--sample`: Number of examples to sample from dataset
- `--difficulty`: Filter by problem difficulty (easy/medium/hard)
- `--start-index`: Start processing from specific index
- `--end-index`: End processing at specific index

### Execution Settings
- `--max-concurrent`: Maximum number of concurrent problems (default: 64)
- `--best-of`: Number of attempts per problem (default: 1)
- `--completions`: Number of completions to try per path (default: 20)
- `--timeout`: Timeout in seconds for code execution (default: 10)
- `--model-timeout`: Timeout for model responses in seconds
- `--tolerance`: Tolerance for numeric answer comparison (default: 1e-2)
- `--retry-count`: Number of retries for failed model calls
- `--cache`: Enable caching of model responses for efficiency

### Output Settings
- `--produce-statistics`: Generate detailed statistics file with component breakdowns
- `--create-dataset`: Create a HuggingFace dataset from results
- `--upload-dataset`: Upload the created dataset to HuggingFace Hub
- `--output-dir`: Directory for saving results and statistics
- `--result-format`: Format for result files (json/csv/both)
- `--verbose`: Verbosity level for console output
- `--log-file`: Path to log file for detailed logging
- `--report-format`: Format for final report (text/html/markdown)

### Benchmark-Specific Settings
- `--similarity-threshold`: Threshold for solution similarity in group evaluation
- `--embedding-model`: Model to use for embedding generation in similarity checks
- `--step-analysis`: Enable detailed step-by-step analysis
- `--code-quality-check`: Enable code quality evaluation
- `--error-analysis`: Enable detailed error analysis and categorization
- `--comparative-mode`: Enable comparison between multiple models
- `--interactive-mode`: Enable interactive evaluation with human feedback

## Example Usage

```bash
# Standard benchmark with a local model and detailed statistics
python -m benchmarks.standard_benchmark --main LOCAL_0 --main-port 8000 --main-temp 0.9 --dataset Metaskepsis/Numina --split validation --produce-statistics --best-of 3 --tolerance 1e-3

# Programming benchmark with cloud models and result dataset creation
python -m benchmarks.programming_benchmark --main GPT --auxiliary CLAUDE --dataset Metaskepsis/Numina --best-of 3 --max-concurrent 32 --timeout 30 --create-dataset --output-dir results/programming_benchmark

# Tutor benchmark with specific problem sources and difficulty filtering
python -m benchmarks.tutor_benchmark --main LOCAL_0 --main-port 8000 --create-dataset --produce-statistics --source AIME,IMO --difficulty medium,hard --sample 100 --seed 42

# Architect benchmark with comparative analysis between different model combinations
python -m benchmarks.architect_benchmark --main GPT --auxiliary CLAUDE --auxiliary2 GEMINI --comparative-mode --dataset Metaskepsis/complex_problems --max-concurrent 16 --report-format html

# Step benchmark with detailed error analysis and visualization
python -m benchmarks.step_benchmark --main LOCAL_0 --main-port 8000 --step-analysis --error-analysis --produce-statistics --output-dir results/step_analysis --report-format markdown
```

## Output and Analysis

Benchmark results are saved in the following formats:
- JSON files with detailed results for each problem including full model responses
- Statistics summary in CSV format with aggregated metrics and breakdowns
- HuggingFace datasets for further analysis or training with standardized schema
- Console output with progress and summary statistics for quick assessment
- HTML/Markdown reports with visualizations and comparative analysis
- Log files with detailed execution traces for debugging

These results can be used to:
1. Evaluate model performance on mathematical reasoning with fine-grained metrics
2. Generate training data for GRPO fine-tuning with targeted examples
3. Identify specific areas for model improvement with error pattern analysis
4. Compare different model architectures and configurations with statistical significance
5. Track improvement over time with consistent evaluation methodology
6. Identify dataset biases and limitations through error analysis
7. Generate insights for curriculum development in mathematical education

## Advanced Features

### Comparative Benchmarking
Compare multiple models on the same problems:
```bash
python -m benchmarks.standard_benchmark --main LOCAL_0 --auxiliary GPT --auxiliary2 CLAUDE --comparative-mode
```

### Incremental Testing
Continue from previous benchmark runs:
```bash
python -m benchmarks.programming_benchmark --continue-from results/previous_run.json
```

### Custom Evaluation Metrics
Add specialized metrics for specific problem types:
```bash
python -m benchmarks.standard_benchmark --custom-metrics geometry,algebra,calculus
```

### Interactive Evaluation
Enable human feedback during benchmark runs:
```bash
python -m benchmarks.tutor_benchmark --interactive-mode --feedback-interval 10
```

### Visualization Generation
Create detailed visualizations of benchmark results:
```bash
python -m benchmarks.step_benchmark --visualize --chart-types histogram,heatmap,scatter
```
