# Technical Context: Mathematical Problem-Solving Framework

**Technologies Used:**

*   **Python 3.8+:** The primary programming language for the framework.
*   **PyTorch 2.0+:** Used for building and training deep learning models.
*   **Transformers:** HuggingFace library for working with pre-trained language models.
*   **Unsloth:** Library for accelerating the training of Qwen models.
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

**Development Setup:**

*   **Python Environment:** Requires a Python 3.8 or later environment.
*   **Dependencies:** Install required libraries using pip (specified in `setup.txt` or similar, though not explicitly shown, it's a standard practice). Key dependencies include PyTorch, Transformers, Unsloth, Datasets, Sympy, NLTK, Sentence-Transformers, and Wandb.
*   **API Keys:** Requires API keys for accessing cloud models (OpenRouter) and HuggingFace Hub (for dataset/model uploads/downloads). These are typically set as environment variables (`OPENROUTER_API_KEY`, `HUGGINGFACE_TOKEN`).
*   **Local Models:** For using local models, requires setting up and running local model endpoints and specifying their ports in the configuration.
*   **Wandb Account:** Requires a Wandb account and API key (`WANDB_API_KEY`) for experiment tracking.

**Technical Constraints:**

*   **Model Size and Memory:** Training large language models can be memory-intensive, addressed by using LoRA and Unsloth.
*   **Computational Resources:** Benchmarking and training can be computationally expensive, addressed by supporting parallel processing and distributed training.
*   **API Rate Limits and Costs:** Interacting with cloud models is subject to API rate limits and costs, addressed by using retry logic, caching, and supporting local models.
*   **Secure Code Execution:** Executing arbitrary code requires a secure sandbox environment to prevent security vulnerabilities.
*   **Data Volume:** Handling large datasets requires efficient processing and storage solutions.
*   **Dependency Management:** Ensuring compatibility between various libraries and their versions.

**Dependencies:**

*   Python 3.8+
*   PyTorch 2.0+
*   Transformers
*   Unsloth
*   Datasets
*   Sympy
*   NLTK
*   Sentence-Transformers
*   Wandb
*   OpenRouter API access (optional, for cloud models)
*   HuggingFace Hub access (optional, for dataset/model sharing)
