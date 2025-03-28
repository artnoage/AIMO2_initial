# Utility Modules for Mathematical Problem Solving

This directory contains utility modules that form the foundation of the mathematical problem-solving framework, supporting both the benchmarking and training components.

## Core Utilities

### `__init__.py`
Sets up the project root path and ensures it's in the Python path for imports.
- Defines `project_root` for consistent file path handling across the project
- Used by both benchmarks and GRPO training scripts

### `agents.py`
Defines agent classes for different mathematical problem-solving tasks:
- `FullSolutionAgent`: Provides complete solutions with analysis and steps
  - Used by `benchmarks.standard_benchmark`
  - Trained by `grpo.solution_qwen0`

- `ProgrammingAgent`: Generates Python code to solve mathematical problems
  - Used by `benchmarks.programming_benchmark`
  - Trained by `grpo.programming_qwen0`

- `TestingAgent`: Creates test functions for mathematical problems
  - Used by `benchmarks.test_benchmark`
  - Trained by `grpo.test_programming_qwen0`

- `TutorAgent`: Evaluates solutions and identifies errors
  - Used by `benchmarks.tutor_benchmark`
  - Trained by `grpo.tutor_grpo`

- `ArchitectAgent`: Analyzes problems and creates prompts for programming agents
  - Used by `benchmarks.architect_benchmark`
  - Trained by `grpo.dynamic_qwen0` (as part of multi-task training)

- `FinalizationAgent`: Completes partial solutions
  - Used by internal validation processes
  - Trained by `grpo.finalization_grpo`

Each agent implements a specific mathematical problem-solving capability and is used by both benchmarks for evaluation and GRPO scripts for training.

### `benchmark_config.py`
Provides configuration classes and command-line argument parsing for benchmarks:
- `ModelOption`: Enum of available models (local and API-based)
- `BenchmarkConfig`: Configuration dataclass with settings for benchmarks
- Used by all benchmark scripts to ensure consistent configuration

### `data_preparation.py`
Functions for preparing training data from datasets:
- `prepare_solution_data`: Formats data for standard mathematical solutions
- `prepare_programming_data`: Formats data for Python code generation
- `prepare_test_programming_data`: Formats data for test function creation
- `prepare_architect_data`: Formats data for architectural analysis
- `prepare_tutor_data`: Formats data for error identification
- `prepare_finalization_data`: Formats data for completing partial solutions
- `prepare_combined_data`: Combines multiple task types with specified distribution

These functions are used by GRPO training scripts to format datasets appropriately for each task.

### `logger.py`
Simple logging utility for benchmark runs:
- `BenchmarkLogger`: Accumulates and manages log messages
- Provides consistent logging format across all benchmarks
- Supports both console output and file logging

### `model_utils.py`
Utilities for working with language models:
- `OpenRouterChat`: Interface to OpenRouter API for cloud models
- `CustomChat`/`CustomChat2`: Interfaces to local model endpoints
- `get_model`: Factory function that creates appropriate model interface
- `get_model_response`: Handles model responses with retry and timeout
- `time_limit`: Context manager for enforcing timeouts
- `async_retry`: Decorator for automatic retrying of failed requests

These utilities provide a consistent interface for both local and cloud models, used by both benchmarks and GRPO training.

### `progress_tracker.py`
Tracks and reports progress during benchmark runs:
- `ProgressTracker`: Manages results, statistics, and reporting
- `run_benchmark`: Asynchronous execution of benchmark tasks
- Handles dataset creation and result saving
- Provides real-time progress updates during long benchmark runs

### `similarity_checker.py`
Computes semantic similarity between solutions:
- `SolutionSimilarityChecker`: Uses embedding models to compare solutions
- Used by `grpo.dynamic_reward` to measure solution diversity
- Supports reward bonuses for diverse solutions
- Handles batching and device management for efficient processing

### `solution_utils.py`
Utilities for working with mathematical solutions:
- `extract_numeric_answer`: Extracts and validates numeric answers from LaTeX
- `extract_answer_from_solution`: Finds boxed answers in LaTeX solutions
- `extract_code_from_response`: Extracts Python code from text responses
- `run_code_safely`: Executes Python code with safety constraints
- `validate_solution`: Checks solution structure and step coherence
- `NumericVerifier`: Verifies numeric answers with configurable tolerance
- `run_test_function`: Tests solution verification functions
- `split_into_steps`: Breaks solutions into individual reasoning steps

These utilities form the core validation logic used by both benchmarks and reward functions.

## Integration with Other Components

### Connection to Benchmarks
Utility modules provide the foundation for all benchmark scripts:
- `agents.py` → Defines the agent interfaces used by benchmarks
- `benchmark_config.py` → Configures benchmark execution
- `model_utils.py` → Handles model interaction
- `progress_tracker.py` → Manages benchmark execution and results
- `solution_utils.py` → Validates solutions and answers

### Connection to GRPO Training
Utility modules support the GRPO training process:
- `data_preparation.py` → Formats datasets for training
- `similarity_checker.py` → Measures solution diversity for rewards
- `solution_utils.py` → Validates solutions for reward calculation

### Connection to Auxiliary Tools
Utility modules are used by auxiliary scripts:
- `solution_utils.py` → Used by `auxilary.filter_dataset` for answer validation
- `model_utils.py` → Used by various auxiliary scripts for model interaction

## Usage

Most utility modules are designed to be imported and used by benchmark scripts and GRPO training scripts rather than run directly. They provide the foundation for the entire mathematical problem-solving framework.

Example imports:

```python
# In benchmark scripts
from utils.agents import FullSolutionAgent
from utils.model_utils import get_model
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker

# In GRPO training scripts
from utils.data_preparation import prepare_solution_data
from utils.similarity_checker import SolutionSimilarityChecker
from utils.solution_utils import extract_numeric_answer, validate_solution
```

## Configuration

Many utilities read environment variables for API keys and other settings:
- `OPENROUTER_API_KEY`: Required for accessing OpenRouter API models
- `HUGGINGFACE_TOKEN`: Used for dataset and model uploads/downloads

Local model endpoints are configured through the `BenchmarkConfig` class, typically via command-line arguments to benchmark scripts.

## Extension Points

The utility modules are designed to be extensible:
- New agent types can be added to `agents.py`
- New model interfaces can be added to `model_utils.py`
- Additional data preparation functions can be added to `data_preparation.py`
- New solution validation methods can be added to `solution_utils.py`
