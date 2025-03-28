# Utility Modules for Mathematical Problem Solving

This directory contains utility modules that support the mathematical problem-solving benchmarks and training processes.

## Core Utilities

### `__init__.py`
Sets up the project root path and ensures it's in the Python path for imports.

### `agents.py`
Defines agent classes for different mathematical problem-solving tasks:
- `FinalizationAgent`: Completes partial solutions
- `FullSolutionAgent`: Provides complete solutions with analysis and steps
- `TutorAgent`: Evaluates solutions and identifies errors
- `ProgrammingAgent`: Generates Python code to solve mathematical problems
- `ArchitectAgent`: Analyzes problems and creates prompts for programming agents
- `TestingAgent`: Creates test functions for mathematical problems

### `benchmark_config.py`
Provides configuration classes and command-line argument parsing for benchmarks:
- `ModelOption`: Enum of available models (local and API-based)
- `BenchmarkConfig`: Configuration dataclass with settings for benchmarks

### `data_preparation.py`
Functions for preparing training data from datasets:
- Converts raw problems into various training formats
- Supports solution, programming, test, architect, tutor, and finalization tasks
- Handles data validation and filtering

### `logger.py`
Simple logging utility for benchmark runs:
- `BenchmarkLogger`: Accumulates and manages log messages

### `model_utils.py`
Utilities for working with language models:
- `OpenRouterChat`: Interface to OpenRouter API
- `CustomChat`/`CustomChat2`: Interfaces to local model endpoints
- Helper functions for model initialization and response handling
- Timeout and retry mechanisms

### `progress_tracker.py`
Tracks and reports progress during benchmark runs:
- `ProgressTracker`: Manages results, statistics, and reporting
- Handles dataset creation and result saving

### `similarity_checker.py`
Computes semantic similarity between solutions:
- `SolutionSimilarityChecker`: Uses embedding models to compare solutions
- Handles batching and device management for efficient processing

### `solution_utils.py`
Utilities for working with mathematical solutions:
- Extracts and validates answers from LaTeX
- Parses and manipulates solution steps
- Executes and verifies Python code
- Generates test cases for solution verification

## Usage

Most utility modules are designed to be imported and used by benchmark scripts rather than run directly. They provide the foundation for the benchmark system in the `benchmarks` directory.

Example import:
```python
from utils.agents import FullSolutionAgent
from utils.model_utils import get_model
from utils.benchmark_config import BenchmarkConfig
```

## Configuration

Many utilities read environment variables for API keys and other settings:
- `OPENROUTER_API_KEY`: Required for accessing OpenRouter API models

Local model endpoints are configured through the `BenchmarkConfig` class, typically via command-line arguments to benchmark scripts.
