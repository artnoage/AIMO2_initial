# Math Problem Solving with LLMs

This project implements different approaches to solve mathematical problems using Large Language Models (LLMs), specifically targeting olympiad-style mathematics problems.

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

## Usage

### TIR Generator

```bash
python TIR_generator.py --model MODEL_NAME
```

Available models: CLAUDE, GEMINI_PRO, GPT, NOUS, etc.

### Monte Carlo Solver

```bash
python monte_carlo.py --solver SOLVER_MODEL --judge JUDGE_MODEL
```

Or use the same model for both:
```bash
python monte_carlo.py --both MODEL_NAME
```

## Results

Results are saved in:
- `TIR_data/` for TIR generator output
- `monte_carlo/` for Monte Carlo solver output

Each run produces:
- JSON file with structured data
- Markdown file with human-readable results
- Detailed statistics and performance metrics

## Dependencies

Required Python packages:
- langchain
- openai
- datasets
- dotenv
- tiktoken
- langgraph (for Monte Carlo solver)

## Environment Setup

1. Create a `.env` file with your API keys:
```
OPENROUTER_API_KEY=your_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
