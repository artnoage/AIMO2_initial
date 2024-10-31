# Math Problem Solving with LLMs

This project implements multiple approaches to solve mathematical problems using Large Language Models (LLMs), specifically targeting olympiad-style mathematics problems.

## Components

### 1. TIR Generator (`TIR_generator.py`)

The Test Implementation Record (TIR) generator creates Python implementations of mathematical problem solutions.

**What it does:**
- Loads olympiad problems from the NuminaMath-CoT dataset
- Uses an LLM to generate Python code that solves each problem
- Executes the generated code and compares results with ground truth
- Saves results in both JSON and Markdown formats

**Key features:**
- Supports multiple LLM providers through OpenRouter
- Safe code execution in isolated namespace
- Detailed logging and error handling
- Results saved in human-readable formats

### 2. Monte Carlo Solver (`monte_carlo.py`)

A probabilistic approach to problem-solving using multiple attempts and consensus.

**What it does:**
- Makes multiple solution attempts (up to 20) for each problem
- Uses a judge LLM to evaluate all solutions and determine the most likely correct answer
- Tracks whether the correct answer appeared in any attempt
- Provides detailed performance analysis of both solver and judge

**Key features:**
- Two-agent system (solver and judge)
- Retry mechanism with delay for API failures
- Token usage tracking
- Comprehensive result logging

### 3. Back-and-Forth Solver (`back_and_forth.py`)

An iterative approach using solver and verifier agents that work together.

**What it does:**
- Initial solution attempt by solver agent
- Verification and feedback by verifier agent
- Multiple iterations of refinement based on feedback
- Stops when correct answer found or max iterations reached

**Key features:**
- Two-agent iterative system
- Detailed state tracking
- Conversation history management
- Performance metrics per iteration

### 4. Benchmark System (`benchmark_numina.py`)

A comprehensive benchmarking system for evaluating model performance.

**What it does:**
- Runs models against NuminaMath-CoT dataset
- Tracks accuracy and token usage
- Handles multiple model providers
- Saves detailed benchmark results

**Key features:**
- Support for multiple LLM providers
- Token usage tracking
- Detailed results logging
- Shuffled dataset processing

### 5. Utility Components

#### Librarian Module (`librarian.py`)
- Handles conversation formatting and storage
- Manages markdown file generation
- Provides text block formatting utilities
- Maintains consistent logging format

#### Dataset Tools
- `sample_dataset.py`: Dataset sampling and processing utilities
- `evaluation_pipeline.py`: Evaluation workflow management
- Results stored in `evaluation_results.jsonl`

## Usage

### TIR Generator
```bash
python TIR_generator.py --model MODEL_NAME
```

### Monte Carlo Solver
```bash
python monte_carlo.py --solver SOLVER_MODEL --judge JUDGE_MODEL
# Or use same model for both:
python monte_carlo.py --both MODEL_NAME
```

### Back-and-Forth Solver
```bash
python back_and_forth.py --solver SOLVER_MODEL --verifier VERIFIER_MODEL
# Or use same model for both:
python back_and_forth.py --both MODEL_NAME
```

### Benchmark System
```bash
python benchmark_numina.py --model MODEL_NAME [--split SPLIT_NAME]
```

Available models for all components:
- CLAUDE (Claude 3.5 Sonnet)
- GEMINI_PRO (Gemini Pro 1.5)
- GEMINI_FLASH (Gemini Flash 1.5)
- GPT (GPT-4)
- MASTER (OpenAI Preview)
- LOCAL (Local Mathstral-7B)
- NOUS (Hermes 3 LLaMA)
- And more via OpenRouter

## Results Directory Structure

```
project/
├── TIR_data/           # TIR generator results
├── monte_carlo/        # Monte Carlo solver results
├── conversations/      # Back-and-forth solver logs
├── benchmark_results/  # Benchmark results
└── evaluation_results.jsonl
```

Each results directory contains:
- JSON files with structured data
- Markdown files with human-readable results
- Detailed statistics and performance metrics
- Token usage analytics

## Dependencies

Required Python packages:
- langchain & langgraph
- openai
- datasets
- python-dotenv
- tiktoken
- tqdm
- Additional requirements in requirements.txt

## Environment Setup

1. Create a `.env` file with your API keys:
```
OPENROUTER_API_KEY=your_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development Guidelines

- Use retry decorators for API calls
- Implement proper error handling
- Track token usage for cost optimization
- Save results in both JSON and Markdown formats
- Maintain consistent logging formats
