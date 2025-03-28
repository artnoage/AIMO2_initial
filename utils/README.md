# Utility Modules for Mathematical Problem Solving

This directory contains utility modules that form the foundation of the mathematical problem-solving framework, supporting both the benchmarking and training components.

## Core Utilities

### `__init__.py`
Sets up the project root path and ensures it's in the Python path for imports.
- Defines `project_root` for consistent file path handling across the project
- Used by both benchmarks and GRPO training scripts
- Ensures consistent import paths regardless of execution context
- Initializes basic logging configuration

### `agents.py`
Defines agent classes for different mathematical problem-solving tasks with specialized prompting:

- `FullSolutionAgent`: Provides complete solutions with analysis and steps
  - Used by `benchmarks.standard_benchmark`
  - Trained by `grpo.solution_qwen0`
  - Implements step-by-step reasoning with answer verification
  - Supports LaTeX formatting for mathematical notation
  - Handles different problem types with specialized approaches

- `ProgrammingAgent`: Generates Python code to solve mathematical problems
  - Used by `benchmarks.programming_benchmark`
  - Trained by `grpo.programming_qwen0`
  - Implements secure code generation with execution verification
  - Supports multiple solution approaches with efficiency considerations
  - Handles edge cases and numerical precision issues

- `TestingAgent`: Creates test functions for mathematical problems
  - Used by `benchmarks.test_benchmark`
  - Trained by `grpo.test_programming_qwen0`
  - Generates comprehensive test cases with edge case coverage
  - Implements robust verification logic with tolerance handling
  - Supports both numeric and symbolic verification approaches

- `TutorAgent`: Evaluates solutions and identifies errors
  - Used by `benchmarks.tutor_benchmark`
  - Trained by `grpo.tutor_grpo`
  - Provides constructive feedback with clear explanations
  - Identifies conceptual, computational, and logical errors
  - Implements pedagogical approaches for error correction

- `ArchitectAgent`: Analyzes problems and creates prompts for programming agents
  - Used by `benchmarks.architect_benchmark`
  - Trained by `grpo.dynamic_qwen0` (as part of multi-task training)
  - Decomposes complex problems into manageable components
  - Provides implementation guidance with algorithm selection
  - Identifies potential pitfalls and edge cases

- `FinalizationAgent`: Completes partial solutions
  - Used by internal validation processes
  - Trained by `grpo.finalization_grpo`
  - Continues from partial solutions with consistent reasoning
  - Maintains coherence with provided steps
  - Verifies final answers against expected results

Each agent implements a specific mathematical problem-solving capability with:
- Carefully crafted system prompts optimized for each task
- Consistent input/output interfaces for interchangeability
- Specialized handling of different problem types and formats
- Detailed logging of reasoning processes
- Error handling and recovery mechanisms

### `benchmark_config.py`
Provides configuration classes and command-line argument parsing for benchmarks:
- `ModelOption`: Enum of available models (local and API-based) with consistent interface
- `BenchmarkConfig`: Configuration dataclass with settings for benchmarks
  - Comprehensive command-line argument parsing with sensible defaults
  - Validation of configuration parameters for consistency
  - Support for configuration from files and environment variables
  - Specialized configurations for different benchmark types
  - Documentation of all configuration options

### `data_preparation.py`
Functions for preparing training data from datasets with task-specific formatting:
- `prepare_solution_data`: Formats data for standard mathematical solutions
  - Structures prompts with system instructions and problem statements
  - Formats expected answers in consistent notation
  - Implements filtering for high-quality examples
  - Supports curriculum learning with difficulty progression

- `prepare_programming_data`: Formats data for Python code generation
  - Structures prompts with programming-specific instructions
  - Includes expected outputs for verification
  - Implements code quality guidelines in prompts
  - Supports different programming approaches

- `prepare_test_programming_data`: Formats data for test function creation
  - Structures prompts with test-specific requirements
  - Includes expected answers for verification function creation
  - Implements comprehensive test case guidelines
  - Supports different testing approaches

- `prepare_architect_data`: Formats data for architectural analysis
  - Structures prompts with decomposition instructions
  - Includes complexity analysis requirements
  - Implements guidance for implementation planning
  - Supports different architectural patterns

- `prepare_tutor_data`: Formats data for error identification
  - Creates examples with deliberate errors for identification
  - Includes correct solutions for comparison
  - Implements pedagogical guidelines for feedback
  - Supports different error types and severities

- `prepare_finalization_data`: Formats data for completing partial solutions
  - Creates examples with partial solutions at different stages
  - Includes complete solutions for verification
  - Implements coherence requirements for continuation
  - Supports different completion points

- `prepare_combined_data`: Combines multiple task types with specified distribution
  - Balances different task types with configurable ratios
  - Implements consistent formatting across tasks
  - Supports curriculum learning across task types
  - Provides detailed statistics on dataset composition

These functions implement:
- Token counting for efficient sequence length management
- Quality filtering with configurable criteria
- Consistent formatting across different data sources
- Support for curriculum learning with difficulty progression
- Detailed logging of dataset statistics

### `logger.py`
Comprehensive logging utility for benchmark runs:
- `BenchmarkLogger`: Accumulates and manages log messages
  - Provides consistent logging format across all benchmarks
  - Supports both console output and file logging
  - Implements different verbosity levels
  - Handles structured logging with component categorization
  - Supports timed logging for performance analysis
  - Implements color coding for different message types
  - Provides progress indicators for long-running operations

### `model_utils.py`
Utilities for working with language models with robust error handling:
- `OpenRouterChat`: Interface to OpenRouter API for cloud models
  - Handles authentication and rate limiting
  - Implements configurable retry logic
  - Supports different model providers through OpenRouter
  - Handles response parsing and error recovery

- `CustomChat`/`CustomChat2`: Interfaces to local model endpoints
  - Supports different API formats (OpenAI-compatible, vLLM, etc.)
  - Implements connection pooling for efficiency
  - Handles streaming responses with timeout protection
  - Supports different prompt formats and tokenization

- `get_model`: Factory function that creates appropriate model interface
  - Selects appropriate implementation based on configuration
  - Handles fallback mechanisms for unavailable models
  - Implements consistent interface across model types
  - Supports model-specific parameter configuration

- `get_model_response`: Handles model responses with retry and timeout
  - Implements exponential backoff for retries
  - Handles different error types with appropriate recovery
  - Supports response validation and formatting
  - Provides detailed error information for debugging

- `time_limit`: Context manager for enforcing timeouts
  - Implements cross-platform timeout handling
  - Supports nested timeouts with priority handling
  - Provides clean resource management with signal handling
  - Implements graceful termination with resource cleanup

- `async_retry`: Decorator for automatic retrying of failed requests
  - Supports configurable retry counts and delays
  - Implements exponential backoff with jitter
  - Handles different exception types with custom recovery
  - Provides detailed logging of retry attempts

These utilities provide:
- A consistent interface for both local and cloud models
- Robust error handling and recovery mechanisms
- Efficient resource management with connection pooling
- Detailed logging for debugging and performance analysis
- Support for different API formats and protocols

### `progress_tracker.py`
Comprehensive tracking and reporting for benchmark runs:
- `ProgressTracker`: Manages results, statistics, and reporting
  - Tracks overall progress with ETA estimation
  - Maintains detailed statistics on success rates and performance
  - Handles result aggregation and summarization
  - Supports different output formats (JSON, CSV, HuggingFace datasets)
  - Implements checkpoint saving for long-running benchmarks
  - Provides real-time console updates with progress bars

- `run_benchmark`: Asynchronous execution of benchmark tasks
  - Implements parallel processing with configurable concurrency
  - Handles task scheduling and resource management
  - Supports prioritization of examples based on criteria
  - Implements graceful shutdown with result saving
  - Provides detailed timing information for performance analysis

The progress tracker implements:
- Dataset creation and result saving with consistent formats
- Real-time progress updates during long benchmark runs
- Detailed statistics generation with component breakdowns
- Support for resuming interrupted benchmark runs
- Visualization of progress and results

### `similarity_checker.py`
Computes semantic similarity between solutions with efficient processing:
- `SolutionSimilarityChecker`: Uses embedding models to compare solutions
  - Supports different embedding models with consistent interface
  - Implements efficient batching for large solution sets
  - Handles caching of embeddings for performance
  - Supports different similarity metrics (cosine, dot product, etc.)
  - Implements clustering for solution group analysis
  - Provides visualization of solution similarity

Used for:
- Measuring solution diversity in `grpo.dynamic_reward`
- Supporting reward bonuses for diverse solutions
- Analyzing solution patterns across different models
- Identifying duplicate or highly similar solutions
- Clustering solutions by approach or methodology

### `solution_utils.py`
Comprehensive utilities for working with mathematical solutions:
- `extract_numeric_answer`: Extracts and validates numeric answers from LaTeX
  - Supports different LaTeX notations and formats
  - Handles fractions, decimals, scientific notation, and complex numbers
  - Implements symbolic evaluation with sympy
  - Supports different answer formats (exact, approximate, interval)
  - Handles units and dimensional analysis

- `extract_answer_from_solution`: Finds boxed answers in LaTeX solutions
  - Supports different boxing notations (\boxed{}, ####, etc.)
  - Implements robust parsing with nested structure handling
  - Handles multiple answer formats with consistent extraction
  - Supports answer verification with expected results

- `extract_code_from_response`: Extracts Python code from text responses
  - Supports different code block formats (markdown, indentation, etc.)
  - Implements syntax validation of extracted code
  - Handles multiple code blocks with context-aware selection
  - Supports code cleaning and formatting

- `run_code_safely`: Executes Python code with comprehensive safety constraints
  - Implements secure execution environment with resource limits
  - Handles timeouts and infinite loops with graceful termination
  - Supports different execution modes (eval, exec, module)
  - Implements whitelisting of allowed modules and functions
  - Provides detailed execution traces for debugging
  - Handles different output formats with consistent parsing

- `validate_solution`: Checks solution structure and step coherence
  - Analyzes logical flow between solution steps
  - Verifies mathematical consistency across steps
  - Implements heuristics for common reasoning errors
  - Supports different solution formats and styles
  - Provides detailed feedback on solution quality

- `NumericVerifier`: Verifies numeric answers with configurable tolerance
  - Supports different comparison modes (absolute, relative, hybrid)
  - Handles special values (infinity, NaN, undefined)
  - Implements unit conversion and dimensional analysis
  - Supports interval answers and inequality constraints
  - Provides detailed explanation of verification results

- `run_test_function`: Tests solution verification functions
  - Generates diverse test cases with edge case coverage
  - Implements secure execution with timeout protection
  - Verifies test function correctness on known answers
  - Supports different test function formats and approaches
  - Provides detailed analysis of test function behavior

- `split_into_steps`: Breaks solutions into individual reasoning steps
  - Identifies logical boundaries between steps
  - Handles different step notation styles
  - Supports nested reasoning with hierarchical structure
  - Implements context tracking across steps
  - Provides step-by-step analysis capabilities

These utilities form the core validation logic used by both benchmarks and reward functions with:
- Comprehensive error handling and recovery mechanisms
- Detailed logging for debugging and analysis
- Support for different mathematical notations and formats
- Efficient processing with caching and optimization
- Robust security measures for code execution

## Integration with Other Components

### Connection to Benchmarks
Utility modules provide the foundation for all benchmark scripts:
- `agents.py` → Defines the agent interfaces used by benchmarks with consistent prompting
- `benchmark_config.py` → Configures benchmark execution with comprehensive options
- `model_utils.py` → Handles model interaction with robust error handling
- `progress_tracker.py` → Manages benchmark execution and results with detailed tracking
- `solution_utils.py` → Validates solutions and answers with comprehensive checks
- `logger.py` → Provides consistent logging across benchmark types
- `similarity_checker.py` → Supports solution diversity analysis in benchmarks

### Connection to GRPO Training
Utility modules support the GRPO training process:
- `data_preparation.py` → Formats datasets for training with task-specific processing
- `similarity_checker.py` → Measures solution diversity for rewards with efficient embedding
- `solution_utils.py` → Validates solutions for reward calculation with comprehensive checks
- `model_utils.py` → Handles model interaction for reward calculation with robust error handling
- `logger.py` → Provides consistent logging for training progress and results

### Connection to Auxiliary Tools
Utility modules are used by auxiliary scripts:
- `solution_utils.py` → Used by `auxilary.filter_dataset` for answer validation
- `model_utils.py` → Used by various auxiliary scripts for model interaction
- `data_preparation.py` → Used by `auxilary.process_dataset` for dataset formatting
- `similarity_checker.py` → Used by `auxilary.create_validation_set` for diverse example selection
- `benchmark_config.py` → Used by various auxiliary scripts for configuration parsing

## Usage

Most utility modules are designed to be imported and used by benchmark scripts and GRPO training scripts rather than run directly. They provide the foundation for the entire mathematical problem-solving framework.

Example imports:

```python
# In benchmark scripts
from utils.agents import FullSolutionAgent
from utils.model_utils import get_model
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.solution_utils import extract_numeric_answer, validate_solution
from utils.logger import BenchmarkLogger
from utils.similarity_checker import SolutionSimilarityChecker

# In GRPO training scripts
from utils.data_preparation import prepare_solution_data, prepare_combined_data
from utils.similarity_checker import SolutionSimilarityChecker
from utils.solution_utils import extract_numeric_answer, validate_solution, run_code_safely
from utils.model_utils import get_model, get_model_response, time_limit
```

## Configuration

Many utilities read environment variables for API keys and other settings:
- `OPENROUTER_API_KEY`: Required for accessing OpenRouter API models
- `HUGGINGFACE_TOKEN`: Used for dataset and model uploads/downloads
- `WANDB_API_KEY`: Used for experiment tracking in training
- `LOG_LEVEL`: Controls logging verbosity
- `CACHE_DIR`: Specifies location for cached embeddings and model responses
- `MAX_WORKERS`: Controls parallelism in various operations
- `TIMEOUT_SECONDS`: Default timeout for model calls and code execution

Local model endpoints are configured through the `BenchmarkConfig` class, typically via command-line arguments to benchmark scripts.

## Extension Points

The utility modules are designed to be extensible with clear interfaces:

- New agent types can be added to `agents.py`:
  ```python
  class NewTaskAgent:
      """Agent for new mathematical task type"""
      def __init__(self, model, config):
          self.model = model
          self.config = config
          self.system_prompt = "Specialized prompt for new task..."
      
      async def solve(self, problem, **kwargs):
          """Implement solution approach for new task"""
          # Implementation details
  ```

- New model interfaces can be added to `model_utils.py`:
  ```python
  class NewModelInterface:
      """Interface for new model API"""
      def __init__(self, endpoint, api_key, **kwargs):
          # Setup connection and authentication
      
      async def ainvoke(self, prompt, **kwargs):
          """Implement API call with standard interface"""
          # Implementation details
  ```

- Additional data preparation functions can be added to `data_preparation.py`:
  ```python
  def prepare_new_task_data(data: Dataset, system_prompt: str) -> Dataset:
      """Format dataset for new task type"""
      # Implementation details
  ```

- New solution validation methods can be added to `solution_utils.py`:
  ```python
  def validate_new_task_solution(solution: str, expected: str) -> Tuple[bool, str]:
      """Validate solutions for new task type"""
      # Implementation details
  ```

## Advanced Features

### Embedding Cache Management
Manage embedding caches for efficient similarity checking:
```python
from utils.similarity_checker import SolutionSimilarityChecker

# Create checker with cache
checker = SolutionSimilarityChecker(cache_dir="embeddings_cache")

# Clear cache for specific model
checker.clear_cache(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Custom Logging Configuration
Configure logging with custom handlers and formatters:
```python
from utils.logger import BenchmarkLogger

# Create logger with custom configuration
logger = BenchmarkLogger(
    name="custom_benchmark",
    log_file="logs/custom_run.log",
    console_level="INFO",
    file_level="DEBUG",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Parallel Solution Validation
Process multiple solutions in parallel for efficiency:
```python
from utils.solution_utils import validate_solutions_parallel

# Validate multiple solutions in parallel
results = validate_solutions_parallel(
    solutions=["solution1", "solution2", "solution3"],
    expected_answers=["answer1", "answer2", "answer3"],
    max_workers=4,
    timeout=10
)
```

### Custom Model Configuration
Configure model interfaces with specialized parameters:
```python
from utils.model_utils import get_model, ModelOption

# Get model with custom configuration
model = get_model(
    option=ModelOption.LOCAL_0,
    port=8000,
    temperature=0.7,
    max_tokens=2048,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop_sequences=["```", "Step "],
    timeout=30
)
```

### Dataset Transformation Pipeline
Create custom dataset processing pipelines:
```python
from utils.data_preparation import create_processing_pipeline

# Create and apply custom processing pipeline
pipeline = create_processing_pipeline([
    filter_by_difficulty,
    remove_multiple_choice,
    ensure_boxed_answers,
    tokenize_and_truncate
])

processed_dataset = pipeline(original_dataset)
```
