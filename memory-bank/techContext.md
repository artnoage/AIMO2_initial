# Technical Context: Mathematical Problem-Solving Framework

**Technologies Used:**

*   **Python 3.8+:** The primary programming language for the framework.
*   **PyTorch 2.0+:** Used for building and training deep learning models.
*   **Transformers:** HuggingFace library for working with pre-trained language models.
*   **Unsloth:** Library for accelerating the training of Qwen models (single-GPU only).
*   **TRL (Transformer Reinforcement Learning):** Library for training language models with reinforcement learning techniques, with support for multi-GPU training.
*   **GRPO (Generative Reinforcement Policy Optimization):** Reinforcement learning algorithm for fine-tuning language models.
*   **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning method for large language models.
*   **Accelerate:** HuggingFace library for distributed training across multiple GPUs.
*   **DeepSpeed:** Microsoft library for optimized distributed training with memory efficiency features like ZeRO optimization.
*   **Datasets:** HuggingFace library for managing and processing datasets.
*   **Sympy:** Python library for symbolic mathematics, used for answer evaluation and validation.
*   **NLTK:** Natural Language Toolkit, used for text processing.
*   **Sentence-Transformers:** Library for generating sentence embeddings, used for similarity checking.
*   **Wandb:** Weights & Biases, used for experiment tracking and visualization.
*   **OpenRouter API:** Used for accessing various cloud-based language models.
*   **Local Model Endpoints:** Support for interacting with self-hosted language models (e.g., vLLM).
*   **LaTeX:** Used for representing mathematical expressions and solutions.
*   **JSON, CSV, JSONL:** Supported formats for datasets and results.
*   **Markdown:** Used for documentation and memory bank files.
*   **Mermaid:** Used for generating diagrams in Markdown.
*   **Asyncio:** Python library for asynchronous programming, used for parallel processing of model responses.
*   **Nest-Asyncio:** Library for allowing nested event loops in Jupyter notebooks.

**Development Setup:**

*   **Python Environment:** Requires a Python 3.8 or later environment.
*   **Dependencies:** Install required libraries using pip (specified in `setup.txt` or similar, though not explicitly shown, it's a standard practice). Key dependencies include PyTorch, Transformers, TRL, Unsloth, Accelerate, DeepSpeed, Datasets, Sympy, NLTK, Sentence-Transformers, and Wandb.
*   **API Keys:** Requires API keys for accessing cloud models (OpenRouter) and HuggingFace Hub (for dataset/model uploads/downloads). These are typically set as environment variables (`OPENROUTER_API_KEY`, `HUGGINGFACE_TOKEN`).
*   **Local Models:** For using local models, requires setting up and running local model endpoints and specifying their ports in the configuration.
*   **Wandb Account:** Requires a Wandb account and API key (`WANDB_API_KEY`) for experiment tracking.
*   **GPU Resources:** Training with GRPO requires GPU resources, with memory requirements reduced through the use of LoRA, Unsloth (for single-GPU), or DeepSpeed (for multi-GPU).
*   **Multi-GPU Setup:** For distributed training, requires multiple GPUs and proper configuration of Accelerate or DeepSpeed.
*   **Logging Directory:** Requires a logging directory structure for storing training logs and statistics.

**Technical Constraints:**

*   **Model Size and Memory:** Training large language models can be memory-intensive, addressed by using LoRA, Unsloth (for single-GPU), or DeepSpeed ZeRO optimization (for multi-GPU).
*   **Computational Resources:** Benchmarking and training can be computationally expensive, addressed by supporting parallel processing and distributed training across multiple GPUs.
*   **API Rate Limits and Costs:** Interacting with cloud models is subject to API rate limits and costs, addressed by using retry logic, caching, and supporting local models.
*   **Secure Code Execution:** Executing arbitrary code requires a secure sandbox environment to prevent security vulnerabilities.
*   **Data Volume:** Handling large datasets requires efficient processing and storage solutions.
*   **Dependency Management:** Ensuring compatibility between various libraries and their versions.
*   **Training Stability:** GRPO training can be sensitive to hyperparameters, requiring careful tuning and monitoring.
*   **Reward Function Design:** Designing effective reward functions for different mathematical tasks requires domain expertise and careful implementation.
*   **Distributed Training Complexity:** Multi-GPU training introduces additional complexity in terms of configuration, synchronization, and debugging.

**Dependencies:**

*   Python 3.8+
*   PyTorch 2.0+
*   Transformers
*   TRL (Transformer Reinforcement Learning)
*   Unsloth (for single-GPU training)
*   Accelerate (for multi-GPU training)
*   DeepSpeed (for optimized multi-GPU training)
*   Datasets
*   Sympy
*   NLTK
*   Sentence-Transformers
*   Wandb
*   Asyncio
*   Nest-Asyncio
*   OpenRouter API access (optional, for cloud models)
*   HuggingFace Hub access (optional, for dataset/model sharing)

**GRPO Training Technical Details:**

*   **Reward Functions:** Implemented as Python classes that inherit from a common `BaseReward` abstract base class, with specialized implementations for different mathematical tasks.
*   **Reward Calculation:** Rewards are calculated based on various criteria such as answer correctness, solution quality, code execution, and self-assessment accuracy.
*   **Reward Statistics:** Detailed statistics are tracked during training, including reward distributions, component breakdowns, and task-specific metrics.
*   **Training Configuration:** Training is configured using a `RewardConfig` dataclass that specifies model parameters, reward values, and execution settings.
*   **Model Training Options:**
    * **Single-GPU Training:** Models can be trained using the GRPOTrainer from the TRL library, with integration with Unsloth for efficient training of Qwen models.
    * **Multi-GPU Training:** Models can be trained using the GRPOTrainer from the TRL library with Accelerate or DeepSpeed for distributed training across multiple GPUs.
*   **LoRA Fine-Tuning:** Models are fine-tuned using LoRA adapters, with configurable parameters such as rank, alpha, and target modules.
*   **Logging and Monitoring:** Training progress is monitored using a custom `LoggingCallback` that logs metrics to Wandb and provides detailed summaries.
*   **Model Saving:** Trained models are saved in merged format for easy deployment, with support for quantization and adapter integration.
*   **Reflective Solver:** Specialized reward function for reflective problem-solving that evaluates both solution correctness and self-assessment accuracy.
*   **Distributed Training:** Support for distributed training across multiple GPUs using either Accelerate or DeepSpeed, with configurable options for optimization and memory efficiency.

**Key Technical Implementations:**

*   **Asynchronous Batch Processing:** Reward functions process batches of completions asynchronously using Python's asyncio library.
*   **Secure Code Execution:** Python code generated by models is executed in a secure sandbox environment with timeout protection.
*   **Answer Extraction and Validation:** Mathematical answers are extracted from solutions using regular expressions and validated using numeric comparison with configurable tolerance.
*   **Plurality Voting:** Reward functions implement plurality voting with numerical grouping to identify the most common answer in a batch.
*   **Reflection Metrics:** Reflective solver tracks detailed reflection metrics including true/false positives/negatives for self-assessment accuracy.
*   **Wandb Integration:** Training metrics and reward components are logged to Wandb for visualization and analysis.
*   **Jupyter Compatibility:** Reward functions include special handling for Jupyter notebooks using nest_asyncio.
*   **Multi-GPU Training:** Implementation of distributed training across multiple GPUs using Accelerate and DeepSpeed, with support for gradient checkpointing, mixed precision, and ZeRO optimization.
*   **Model Distribution:** Automatic distribution of model weights across multiple GPUs using device_map="auto" for efficient memory usage.
*   **DeepSpeed Integration:** Configuration of DeepSpeed for optimized distributed training with ZeRO stage 2, CPU offloading, and mixed precision.
