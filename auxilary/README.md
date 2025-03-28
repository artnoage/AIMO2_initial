# Auxiliary Utilities for Mathematical Problem Solving

This directory contains utility scripts for dataset processing, model management, and other supporting functions for the mathematical problem-solving framework. These scripts provide the data preparation and model management foundation for both benchmarking and training.

## Dataset Processing

### `filter_dataset.py`
Filters mathematical problem datasets based on various criteria with configurable thresholds:
- Removes problems with invalid content (HTTP links, non-Latin characters, code blocks)
- Ensures problems have exactly one boxed answer with proper LaTeX formatting
- Validates that answers are numeric with sympy parsing
- Excludes multiple-choice problems (optional) with pattern recognition
- Pushes filtered datasets to HuggingFace Hub (optional) with versioning
- Provides detailed statistics on filtering operations
- Supports incremental filtering with resume capability
- Implements parallel processing for efficiency

Used to prepare high-quality datasets for both benchmarking and training with consistent quality standards.

```bash
# Basic filtering with default settings
python -m auxilary.filter_dataset --repo-name Metaskepsis/Olympiads_hard --output-dir olympiads_filtered

# Advanced filtering with custom parameters
python -m auxilary.filter_dataset --repo-name Metaskepsis/Olympiads_hard --output-dir olympiads_filtered --exclude-multiple-choice --max-length 2000 --push-to-hub-name "username/filtered_olympiads" --token ${HUGGINGFACE_TOKEN} --workers 8
```

### `process_dataset.py`
Processes local datasets to ensure high-quality examples with valid answers and consistent formatting:
- Filters by source (optional) with configurable source list
- Validates boxed answers using `utils.solution_utils.extract_answer_from_solution`
- Checks for invalid content with comprehensive validation rules
- Extracts and verifies numeric answers with sympy integration
- Saves processed datasets in HuggingFace format with metadata
- Implements timeout protection for processing operations
- Provides detailed statistics on processing results
- Supports incremental processing with checkpointing
- Handles different input formats (JSON, CSV, JSONL)

Creates the foundation datasets used by both benchmarks and GRPO training with consistent quality and format.

```bash
# Basic processing with default settings
python -m auxilary.process_dataset --dataset local_datasets/raw_data --output-dir processed_data

# Advanced processing with source filtering and custom parameters
python -m auxilary.process_dataset --dataset local_datasets/raw_data --output-dir processed_data --sources AIME,IMO,Putnam --max-workers 16 --timeout 30 --push-to-hub --hub-repo "username/processed_dataset"
```

### `process_validation_set.py`
Normalizes validation datasets by standardizing field names and ensuring consistent quality:
- Converts 'question' fields to 'problem' fields for consistency across datasets
- Shuffles and selects a subset of examples with stratified sampling
- Uploads processed datasets to HuggingFace Hub with versioning
- Implements field validation and normalization
- Provides detailed statistics on dataset composition
- Supports filtering by difficulty and problem type
- Ensures balanced representation of problem categories
- Implements deduplication with similarity checking

Creates standardized validation sets for consistent model evaluation across different benchmarks.

```bash
# Basic validation set processing
python -m auxilary.process_validation_set --source-repo Metaskepsis/validation_set --target-repo Metaskepsis/validation_set_mini

# Advanced processing with filtering and sampling
python -m auxilary.process_validation_set --source-repo Metaskepsis/validation_set --target-repo Metaskepsis/validation_set_mini --sample-size 500 --difficulty-distribution easy=0.3,medium=0.4,hard=0.3 --categories algebra,geometry,calculus,number_theory --deduplicate
```

### `create_validation_set.py`
Creates validation datasets by combining multiple sources with configurable distribution:
- Merges examples from different datasets with source tracking
- Filters for valid numeric answers with comprehensive validation
- Assigns unique IDs to examples for consistent tracking
- Uploads combined datasets to HuggingFace Hub with metadata
- Implements stratified sampling for balanced representation
- Provides detailed statistics on dataset composition
- Supports filtering by difficulty and problem type
- Ensures diversity with similarity-based selection
- Implements quality checks for all examples

Creates diverse validation sets for comprehensive model evaluation with controlled distribution of problem types and difficulties.

```bash
# Basic validation set creation
python -m auxilary.create_validation_set

# Advanced creation with source specification and distribution control
python -m auxilary.create_validation_set --sources Metaskepsis/Numina,Metaskepsis/Olympiads --output-repo "username/validation_set" --sample-size 500 --difficulty-distribution easy=0.3,medium=0.4,hard=0.3 --deduplicate --min-boxed-answers 1 --max-boxed-answers 1
```

### `merge_json.py`
Merges multiple JSON files containing arrays of objects with comprehensive options:
- Combines all JSON files in a specified folder with configurable patterns
- Outputs a single merged JSON file with optional pretty printing
- Useful for combining benchmark results or dataset fragments
- Implements deduplication with configurable keys
- Provides detailed statistics on merge operations
- Supports filtering by field values
- Implements sorting by specified fields
- Handles different JSON structures with flexible parsing
- Supports output in different formats (JSON, JSONL, CSV)

Used to combine results from multiple benchmark runs or to merge dataset fragments with consistent formatting.

```bash
# Basic JSON merging
python -m auxilary.merge_json results_folder --output merged.json

# Advanced merging with deduplication and filtering
python -m auxilary.merge_json results_folder --output merged.json --deduplicate-key id --filter-field status=success --sort-by timestamp --format jsonl --pretty-print
```

## Model Management

### `export_model.py`
Exports trained models from checkpoints with optimization options:
- Loads base models and applies LoRA adapters with proper configuration
- Merges adapters with base models for deployment efficiency
- Saves in optimized formats (16-bit, 8-bit, or 4-bit quantization)
- Organizes models by type and timestamp for versioning
- Implements validation of exported models
- Provides detailed statistics on model size and parameters
- Supports different adapter configurations
- Implements safeguards for memory-efficient export
- Handles different model architectures with appropriate settings

Converts GRPO-trained models into deployable formats for benchmarking with optimization for inference efficiency.

```bash
# Basic model export
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/tutor_20240315 --output-dir models/tutor

# Advanced export with quantization and validation
python -m auxilary.export_model --model-name unsloth/Qwen1.5-7B --checkpoint checkpoints/tutor_20240315 --output-dir models/tutor --quantize --bits 4 --validate --save-adapter --compression gptq
```

### `sft0.py`
Performs supervised fine-tuning on mathematical problem-solving with comprehensive configuration:
- Configures LoRA for parameter-efficient training with customizable parameters
- Formats conversations with system prompts for different tasks
- Trains on mathematical problem datasets with curriculum learning
- Saves merged models after training with versioning
- Implements evaluation during training
- Provides detailed training statistics and logs
- Supports mixed precision training for efficiency
- Implements gradient accumulation for effective batch size
- Handles different model architectures with appropriate settings
- Supports distributed training across multiple GPUs

Provides initial supervised fine-tuning before GRPO training with comprehensive configuration options.

```bash
# Basic supervised fine-tuning
python -m auxilary.sft0

# Advanced fine-tuning with custom parameters
python -m auxilary.sft0 --model_name unsloth/Qwen1.5-7B --dataset Metaskepsis/Numina --learning_rate 2e-5 --epochs 3 --batch_size 4 --lora_r 32 --lora_alpha 64 --gradient_accumulation_steps 8 --mixed_precision bf16 --output_dir models/sft_math
```

### `hf_loader.py`
Handles uploading and downloading datasets/models to/from HuggingFace Hub with comprehensive options:
- Supports both dataset and model operations with consistent interface
- Validates local content before upload with integrity checks
- Creates appropriate directory structures with metadata
- Handles authentication via API tokens with secure storage
- Implements retry logic for network operations
- Provides detailed progress tracking for large transfers
- Supports versioning and tagging of uploads
- Implements incremental uploads for efficiency
- Handles different dataset formats with appropriate conversion
- Supports private repositories with access control

Used by both benchmark and training scripts to access datasets and models with consistent interface.

```bash
# Download a dataset
python -m auxilary.hf_loader --type dataset --load down --repo_name Metaskepsis/Numina --path local_datasets/numina

# Upload a model with versioning
python -m auxilary.hf_loader --type model --load up --repo_name username/model-name --path models/my_model --version v1.0 --private --commit_message "Initial release" --create_repo

# Download specific dataset split
python -m auxilary.hf_loader --type dataset --load down --repo_name Metaskepsis/Numina --path local_datasets/numina --split validation --revision main
```

### `datatype_transformation.py`
Converts between JSON and Arrow dataset formats with comprehensive options:
- Transforms JSON datasets to HuggingFace Arrow format with schema validation
- Converts Arrow datasets back to JSON with configurable formatting
- Normalizes field names across entries for consistency
- Preserves all fields from source data with metadata
- Implements validation of converted datasets
- Provides detailed statistics on conversion operations
- Supports different JSON structures with flexible parsing
- Handles large datasets with memory-efficient processing
- Implements parallel processing for efficiency
- Supports different output formats (JSON, JSONL, CSV)

Facilitates dataset format conversion for different processing needs with consistent quality and format.

```bash
# JSON to Arrow conversion
python -m auxilary.datatype_transformation dataset.json --type math_problems

# Arrow to JSON conversion with formatting options
python -m auxilary.datatype_transformation local_datasets/math_problems --output dataset.json --format jsonl --pretty-print --fields problem,solution,answer

# Conversion with schema validation
python -m auxilary.datatype_transformation dataset.json --type math_problems --schema schemas/math_problems.json --validate
```

## Integration with Other Components

### Connection to Benchmarks
Auxiliary scripts prepare the datasets used by benchmark scripts:
- `filter_dataset.py` → Creates clean datasets for `benchmarks.*_benchmark.py` with consistent quality
- `create_validation_set.py` → Creates validation sets for consistent evaluation across benchmarks
- `export_model.py` → Prepares models for evaluation with benchmarks in optimized formats
- `process_validation_set.py` → Standardizes validation datasets for consistent benchmarking
- `merge_json.py` → Combines benchmark results for comprehensive analysis

The auxiliary scripts ensure that:
- Benchmarks have access to high-quality, consistent datasets
- Models are properly exported and optimized for evaluation
- Results can be aggregated and analyzed across multiple runs
- Validation sets provide consistent evaluation metrics

### Connection to GRPO Training
Auxiliary scripts support the GRPO training process:
- `process_dataset.py` → Prepares high-quality training data with consistent formatting
- `sft0.py` → Provides initial supervised fine-tuning before GRPO training
- `export_model.py` → Converts trained models to deployable format with optimization
- `filter_dataset.py` → Ensures training data meets quality standards
- `datatype_transformation.py` → Converts datasets to appropriate formats for training

The auxiliary scripts ensure that:
- Training data is high-quality and consistently formatted
- Models are properly initialized with supervised fine-tuning
- Trained models are correctly exported for deployment
- Dataset formats are compatible with training requirements

### Connection to Utilities
Auxiliary scripts use utility modules for core functionality:
- `utils.solution_utils` → Used for answer extraction and validation with comprehensive checks
- `utils.model_utils` → Used for model interaction with robust error handling
- `utils.data_preparation` → Used for dataset formatting with task-specific processing
- `utils.similarity_checker` → Used for deduplication and diversity measurement
- `utils.benchmark_config` → Used for configuration parsing and validation

The auxiliary scripts leverage utility modules to:
- Ensure consistent validation logic across the framework
- Handle model interactions with robust error recovery
- Format datasets consistently for different tasks
- Measure similarity and diversity in datasets
- Parse and validate configuration options

## Workflow Integration

The auxiliary scripts fit into the overall project workflow with clear dependencies and data flow:

1. **Data Preparation**:
   - `process_dataset.py` → Process raw datasets from various sources
   - `filter_dataset.py` → Filter for high-quality examples with consistent criteria
   - `create_validation_set.py` → Create validation datasets for evaluation
   - `datatype_transformation.py` → Convert between dataset formats as needed
   - `process_validation_set.py` → Standardize validation datasets for consistency

2. **Model Training**:
   - `sft0.py` → Initial supervised fine-tuning with curriculum learning
   - GRPO training scripts → Reinforcement learning with specialized rewards
   - `export_model.py` → Export trained models with optimization for deployment
   - `hf_loader.py` → Upload models to HuggingFace Hub for sharing

3. **Evaluation**:
   - Benchmark scripts → Evaluate model performance on validation sets
   - `merge_json.py` → Combine benchmark results for comprehensive analysis
   - Analysis of results to guide further training iterations
   - `hf_loader.py` → Download models and datasets for evaluation

This workflow creates a continuous improvement cycle:
- Data preparation ensures high-quality training and evaluation data
- Model training improves capabilities on specific mathematical tasks
- Evaluation measures performance and identifies areas for improvement
- Results guide refinement of datasets and training approaches

## Common Features

All auxiliary scripts share these common features for consistency and robustness:
- Command-line interfaces with argument parsing and validation
  - Comprehensive help documentation
  - Sensible defaults for common use cases
  - Validation of input parameters
  - Support for configuration files

- Detailed logging of operations with configurable verbosity
  - Progress indicators for long-running operations
  - Error reporting with actionable information
  - Statistics on processing operations
  - Timing information for performance analysis

- Error handling and validation with graceful recovery
  - Comprehensive input validation
  - Graceful handling of network errors
  - Recovery from interrupted operations
  - Detailed error messages for debugging

- Integration with HuggingFace ecosystem for sharing and versioning
  - Dataset and model versioning
  - Metadata management
  - Access control for private resources
  - Consistent interface across different resource types

- Support for local and remote datasets with consistent handling
  - Transparent access to local and remote resources
  - Caching for efficient access
  - Incremental processing for large datasets
  - Format conversion as needed

- Progress tracking for long-running operations with ETA estimation
  - Real-time progress updates
  - Resource usage monitoring
  - ETA estimation for completion
  - Checkpointing for resumable operations

## Advanced Features

### Parallel Processing
Most scripts support parallel processing for efficiency:
```bash
python -m auxilary.filter_dataset --repo-name Metaskepsis/Olympiads_hard --output-dir olympiads_filtered --workers 16
```

### Incremental Processing
Support for resuming interrupted operations:
```bash
python -m auxilary.process_dataset --dataset local_datasets/raw_data --output-dir processed_data --resume --checkpoint-file checkpoint.json
```

### Custom Validation Rules
Define custom validation rules for dataset filtering:
```bash
python -m auxilary.filter_dataset --repo-name Metaskepsis/Olympiads_hard --output-dir olympiads_filtered --validation-rules rules.json
```

### Dataset Transformation Pipeline
Create custom dataset processing pipelines:
```bash
python -m auxilary.process_dataset --dataset local_datasets/raw_data --output-dir processed_data --pipeline filter,normalize,validate,transform
```

### Model Merging
Merge multiple specialized models:
```bash
python -m auxilary.merge_models --models models/solution,models/programming,models/tutor --output models/combined --weights 0.4,0.3,0.3
```
