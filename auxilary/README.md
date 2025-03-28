# Auxiliary Utilities for Mathematical Problem Solving

This directory contains utility scripts for dataset processing, model management, and other supporting functions for the mathematical problem-solving framework. These scripts provide the data preparation and model management foundation for both benchmarking and training.

## Dataset Processing

### `filter_dataset.py`
Filters mathematical problem datasets based on various criteria:
- Removes problems with invalid content (HTTP links, non-Latin characters)
- Ensures problems have exactly one boxed answer
- Validates that answers are numeric
- Excludes multiple-choice problems (optional)
- Pushes filtered datasets to HuggingFace Hub (optional)

Used to prepare high-quality datasets for both benchmarking and training.

```bash
python -m auxilary.filter_dataset --repo-name Metaskepsis/Olympiads_hard --output-dir olympiads_filtered
```

### `process_dataset.py`
Processes local datasets to ensure high-quality examples with valid answers:
- Filters by source (optional)
- Validates boxed answers using `utils.solution_utils.extract_answer_from_solution`
- Checks for invalid content
- Extracts and verifies numeric answers
- Saves processed datasets in HuggingFace format

Creates the foundation datasets used by both benchmarks and GRPO training.

```bash
python -m auxilary.process_dataset --dataset local_datasets/raw_data --output-dir processed_data
```

### `process_validation_set.py`
Normalizes validation datasets by standardizing field names:
- Converts 'question' fields to 'problem' fields for consistency
- Shuffles and selects a subset of examples
- Uploads processed datasets to HuggingFace Hub

Creates standardized validation sets for consistent model evaluation.

```bash
python -m auxilary.process_validation_set --source-repo Metaskepsis/validation_set --target-repo Metaskepsis/validation_set_mini
```

### `create_validation_set.py`
Creates validation datasets by combining multiple sources:
- Merges examples from different datasets
- Filters for valid numeric answers
- Assigns unique IDs to examples
- Uploads combined datasets to HuggingFace Hub

Creates diverse validation sets for comprehensive model evaluation.

```bash
python -m auxilary.create_validation_set
```

### `merge_json.py`
Merges multiple JSON files containing arrays of objects:
- Combines all JSON files in a specified folder
- Outputs a single merged JSON file
- Useful for combining benchmark results or dataset fragments

Used to combine results from multiple benchmark runs or to merge dataset fragments.

```bash
python -m auxilary.merge_json results_folder --output merged.json
```

## Model Management

### `export_model.py`
Exports trained models from checkpoints:
- Loads base models and applies LoRA adapters
- Merges adapters with base models
- Saves in optimized 16-bit format
- Organizes models by type and timestamp

Converts GRPO-trained models into deployable formats for benchmarking.

```bash
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/tutor_20240315 --output-dir models/tutor
```

### `sft0.py`
Performs supervised fine-tuning on mathematical problem-solving:
- Configures LoRA for parameter-efficient training
- Formats conversations with system prompts
- Trains on mathematical problem datasets
- Saves merged models after training

Provides initial supervised fine-tuning before GRPO training.

```bash
python -m auxilary.sft0
```

### `hf_loader.py`
Handles uploading and downloading datasets/models to/from HuggingFace Hub:
- Supports both dataset and model operations
- Validates local content before upload
- Creates appropriate directory structures
- Handles authentication via API tokens

Used by both benchmark and training scripts to access datasets and models.

```bash
# Download a dataset
python -m auxilary.hf_loader --type dataset --load down --repo_name Metaskepsis/Numina --path local_datasets/numina

# Upload a model
python -m auxilary.hf_loader --type model --load up --repo_name username/model-name --path models/my_model
```

### `datatype_transformation.py`
Converts between JSON and Arrow dataset formats:
- Transforms JSON datasets to HuggingFace Arrow format
- Converts Arrow datasets back to JSON
- Normalizes field names across entries
- Preserves all fields from source data

Facilitates dataset format conversion for different processing needs.

```bash
# JSON to Arrow
python -m auxilary.datatype_transformation dataset.json --type math_problems

# Arrow to JSON
python -m auxilary.datatype_transformation local_datasets/math_problems --output dataset.json
```

## Integration with Other Components

### Connection to Benchmarks
Auxiliary scripts prepare the datasets used by benchmark scripts:
- `filter_dataset.py` → Creates clean datasets for `benchmarks.*_benchmark.py`
- `create_validation_set.py` → Creates validation sets for consistent evaluation
- `export_model.py` → Prepares models for evaluation with benchmarks

### Connection to GRPO Training
Auxiliary scripts support the GRPO training process:
- `process_dataset.py` → Prepares high-quality training data
- `sft0.py` → Provides initial supervised fine-tuning
- `export_model.py` → Converts trained models to deployable format

### Connection to Utilities
Auxiliary scripts use utility modules for core functionality:
- `utils.solution_utils` → Used for answer extraction and validation
- `utils.model_utils` → Used for model interaction
- `utils.data_preparation` → Used for dataset formatting

## Workflow Integration

The auxiliary scripts fit into the overall project workflow:

1. **Data Preparation**:
   - `process_dataset.py` → Process raw datasets
   - `filter_dataset.py` → Filter for high-quality examples
   - `create_validation_set.py` → Create validation datasets

2. **Model Training**:
   - `sft0.py` → Initial supervised fine-tuning
   - GRPO training scripts → Reinforcement learning
   - `export_model.py` → Export trained models

3. **Evaluation**:
   - Benchmark scripts → Evaluate model performance
   - `merge_json.py` → Combine benchmark results
   - Analysis of results to guide further training

## Common Features

All auxiliary scripts share these common features:
- Command-line interfaces with argument parsing
- Detailed logging of operations
- Error handling and validation
- Integration with HuggingFace ecosystem
- Support for local and remote datasets
- Progress tracking for long-running operations
