import os
import argparse
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, login
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_info(message):
    """Log information messages"""
    logger.info(message)

def push_to_hub(dataset: Dataset, repo_name: str, token: Optional[str] = None) -> str:
    """
    Push a dataset to the Hugging Face Hub.
    
    Args:
        dataset: The dataset to push
        repo_name: Target repository name (e.g., 'username/dataset_name')
        token: HF API token (optional if already logged in)
        
    Returns:
        URL of the dataset on the Hub
    """
    if token:
        login(token)
    
    log_info(f"Pushing dataset to {repo_name}...")
    api = HfApi()
    
    # Push the dataset to the Hub
    url = dataset.push_to_hub(repo_name)
    
    log_info(f"Dataset successfully pushed to {url}")
    return url

def process_validation_set(
    source_repo: str = "Metaskepsis/validation_set",
    target_repo: str = "Metaskepsis/validation_set_filtered",
    token: Optional[str] = None
):
    """
    Download validation dataset, normalize question/problem fields, and upload the processed dataset.
    
    Args:
        source_repo: Source repository name on Hugging Face
        target_repo: Target repository name for processed dataset
        token: HF API token (optional if already logged in)
    """
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message="Metadata validation was skipped")
    warnings.filterwarnings("ignore", message="Found cached dataset")
    
    try:
        # Load the dataset
        log_info(f"Loading dataset from {source_repo}...")
        
        # Try to load the dataset
        try:
            dataset = load_dataset(source_repo, split="train")
            split_used = "train"
        except Exception:
            # If 'train' fails, try without specifying a split
            try:
                dataset_dict = load_dataset(source_repo)
                # Use the first available split
                split_used = list(dataset_dict.keys())[0]
                dataset = dataset_dict[split_used]
            except Exception as e:
                log_info(f"Error loading dataset: {e}")
                return
        
        log_info(f"Successfully loaded dataset with {len(dataset)} examples from split '{split_used}'")
        
        # Process the dataset
        log_info("Processing dataset to normalize question/problem fields...")
        
        # Create a function to process each example
        def process_example(example):
            processed = dict(example)  # Create a copy of the example
            
            # Check if 'question' exists but 'problem' doesn't
            if 'question' in processed and 'problem' not in processed:
                processed['problem'] = processed['question']
                # Keep the question field for backward compatibility
            
            return processed
        
        # Apply the processing function to each example
        processed_dataset = dataset.map(process_example)
        processed_dataset = processed_dataset.shuffle(seed=171)
    # Use a reasonable number of examples
        processed_dataset = processed_dataset.select(range(100))
        log_info(f"Processing complete. Dataset has {len(processed_dataset)} examples.")
        
        # Push the processed dataset to the Hub
        push_to_hub(processed_dataset, target_repo, token)
        
        log_info("Dataset processing and upload complete!")
        
    except Exception as e:
        log_info(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Process validation dataset to normalize question/problem fields')
    parser.add_argument('--source-repo', type=str, default="Metaskepsis/validation_set",
                       help='Source HuggingFace repository name (default: "Metaskepsis/validation_set")')
    parser.add_argument('--target-repo', type=str, default="Metaskepsis/validation_set_mini",
                       help='Target HuggingFace repository name (default: "Metaskepsis/validation_set_mini")')
    parser.add_argument('--token', type=str, default=None,
                       help='HuggingFace API token (optional if already logged in)')
    
    args = parser.parse_args()
    
    process_validation_set(
        source_repo=args.source_repo,
        target_repo=args.target_repo,
        token=args.token
    )

if __name__ == "__main__":
    main()
