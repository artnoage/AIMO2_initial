import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset, load_from_disk
from tqdm import tqdm
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import  login
import json
import argparse

def validate_arrow_dataset(path: Path) -> bool:
    """Validate that path points to an Arrow dataset directory."""
    if not path.is_dir():
        raise ValueError(f"Path must be a directory: {path}")
    if not any(f.name == 'dataset_info.json' for f in path.glob('*')):
        raise ValueError(f"Directory must contain dataset_info.json: {path}")
    return True

def validate_repo_name(repo_name: str) -> bool:
    """Validate repository name format (username/repo-name)."""
    if not repo_name or '/' not in repo_name:
        return False
    username, repo = repo_name.split('/')
    return bool(username and repo)

def upload_dataset_to_hub(dataset, repo_name):
    """Upload the dataset to HuggingFace Hub."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    if not validate_repo_name(repo_name):
        raise ValueError("Repository name must be in format 'username/repo-name'")
        
    login(token)
    print(f"Uploading dataset to {repo_name}...")
    with tqdm(total=100, desc="Uploading") as pbar:
        dataset.push_to_hub(
            repo_name
        )

def upload_model_to_hub(model_path, repo_name):
    """Upload a model to HuggingFace Hub."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token)
    
    print(f"Loading model and tokenizer from {model_path}...")
    try:
        # Load config to check dtype
        config = AutoConfig.from_pretrained(model_path)
        
        # Determine dtype from config or default to float16
        dtype = torch.float16
        if hasattr(config, 'torch_dtype'):
            if config.torch_dtype == 'float32':
                dtype = torch.float32
            elif config.torch_dtype == 'bfloat16':
                dtype = torch.bfloat16
        
        # Load model and tokenizer using standard paths, preserving dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test tokenizer
        test_text = "Testing the tokenizer"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print(f"Tokenizer test - Original: '{test_text}' -> Decoded: '{decoded}'")
        
        print("Pushing to hub...")
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        print("Upload complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure your model directory contains:")
        print("- config.json")
        print("- pytorch_model.bin (or model.safetensors)")
        print("- tokenizer_config.json")
        print("- tokenizer.json")
        print("- special_tokens_map.json")
        raise

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Upload Arrow datasets or models to HuggingFace Hub')
    parser.add_argument('--type', choices=['dataset', 'model'], required=True,
                      help='Type of content to upload (dataset or model)')
    parser.add_argument('--path', required=True, type=Path,
                      help='Path to Arrow dataset directory or model directory')
    parser.add_argument('--repo_name', required=True,
                      help='Name for the HuggingFace repository (format: username/repo-name)')
    args = parser.parse_args()
    
    if args.type == 'dataset':
        # Validate path exists
        if not args.path.exists():
            raise FileNotFoundError(f"Path does not exist: {args.path}")

        try:
            # Validate and load Arrow dataset
            validate_arrow_dataset(args.path)
            print(f"Loading Arrow dataset from {args.path}...")
            dataset = load_from_disk(str(args.path))
            print("Dataset loaded successfully")
            
            # Upload dataset
            upload_dataset_to_hub(dataset, args.repo_name)
            print(f"Dataset successfully uploaded to {args.repo_name}")
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            raise
    
    else:  # model
        if not args.repo_name:
            raise ValueError("--repo_name is required for model uploads")
            
        # Handle model upload
        upload_model_to_hub(args.path, args.repo_name)
        print(f"Model successfully uploaded to {args.repo_name}")

if __name__ == "__main__":
    main()
