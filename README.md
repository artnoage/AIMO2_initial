# Mathematical Problem Solving Framework

This project implements a framework for solving mathematical problems using various language models, with different approaches and evaluation methods.

## Core Components

### Main Solvers

1. **multiple_sampling.py**
   - Implements Monte Carlo approach to problem solving
   - Uses multiple attempts to solve each problem
   - Aggregates results using a judge model
   - Saves intermediate results every 2000 examples

2. **multiple_turns.py**
   - Implements iterative solving with verification
   - Uses solver and verifier models in turns
   - Maximum 4 iterations per problem
   - Saves detailed conversation logs

3. **benchmark_numina.py**
   - Benchmarks models on the Numina dataset
   - Supports concurrent problem processing
   - Saves intermediate results and augmented datasets
   - Tracks accuracy and performance metrics

4. **synthetic.py**
   - Implements synthetic problem solving
   - Uses partial solutions as hints
   - Saves intermediate results and error rates
   - Creates augmented datasets with model responses

### Utility Modules

1. **utils/augmented_data_handler.py**
   - Manages saving and loading of augmented datasets
   - Handles file existence checks and user preferences
   - Supports appending and replacing data modes

2. **utils/utils.py**
   - Contains shared utility functions
   - Implements answer extraction from LaTeX
   - Handles mathematical notation parsing

3. **utils/filter_dataset.py**
   - Filters the NuminaMath-CoT dataset
   - Creates Numina-Olympiads dataset
   - Ensures valid boxed answers
   - Pushes filtered dataset to HuggingFace

## Supported Models

The framework supports various language models through the ModelOption enum:
- Claude (Anthropic)
- Gemini Pro/Flash (Google)
- GPT-4 (OpenAI)
- Local models (e.g., Mathstral-7B)
- And more...

## Usage

### Basic Usage

```bash
# Run multiple sampling approach
python multiple_sampling.py --solver MODEL --judge MODEL

# Run multiple turns approach
python multiple_turns.py --solver MODEL --verifier MODEL

# Run benchmark
python benchmark_numina.py --solver MODEL --verifier MODEL

# Run synthetic approach
python synthetic.py --solver MODEL --verifier MODEL
```

### Additional Options

- `--both MODEL`: Use same model for solver and verifier/judge
- `--split SPLIT`: Choose dataset split (train/validation/test)
- `--source SOURCE`: Filter problems by source
- `--max-concurrent N`: Set maximum concurrent problems (default: 4)

The framework includes robust error handling:
- Automatic backup of corrupted JSON files
- Graceful handling of parsing errors
- Progress preservation on interruption

## Results

Results are saved in different directories:
- `results/`: All benchmark and synthetic results including:
  - Final results with error rates
  - Intermediate results every 100 examples
  - Error rate progression data
- `augmented_datasets/`: Generated datasets with model responses
  - Includes automatic backup of corrupted files
  - Supports both append and replace modes
- `conversations/`: Detailed solving process logs

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required environment variables:
- `OPENROUTER_API_KEY`: For accessing various LLM APIs
- `SAMBANOVA_API_KEY`: For SambaNova models (optional)

## Dataset

The framework uses the Numina-Olympiads dataset, a filtered version of NuminaMath-CoT containing only olympiad problems with valid boxed answers.

## License

MIT License
