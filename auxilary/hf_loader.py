import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset, load_from_disk, load_dataset
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

def download_dataset_from_hub(repo_name: str, local_path: Path):
    """Download a dataset from HuggingFace Hub."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
        
    login(token)
    print(f"Downloading dataset from {repo_name}...")
    dataset = load_dataset(repo_name)
    
    # Create local_datasets directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to {local_path}...")
    dataset.save_to_disk(str(local_path))
    print("Download complete!")

def download_model_from_hub(repo_name: str, local_path: Path):
    """Download a model from HuggingFace Hub."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token)
    
    print(f"Downloading model and tokenizer from {repo_name}...")
    try:
        # Create models directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Download model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(repo_name)
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        
        print(f"Saving model and tokenizer to {local_path}...")
        model.save_pretrained(str(local_path))
        tokenizer.save_pretrained(str(local_path))
        print("Download complete!")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        raise

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
    
    parser = argparse.ArgumentParser(description='Upload or download datasets/models from HuggingFace Hub')
    parser.add_argument('--type', choices=['dataset', 'model'], required=True,
                      help='Type of content to handle (dataset or model)')
    parser.add_argument('--load', choices=['up', 'down'], required=True,
                      help='Upload to or download from HuggingFace Hub')
    parser.add_argument('--path', type=Path,
                      help='Path to local content (for upload) or destination path (for download)')
    parser.add_argument('--repo_name', required=True,
                      help='Name for the HuggingFace repository (format: username/repo-name)')
    args = parser.parse_args()
    
    if args.load == 'up':
        if not args.path:
            raise ValueError("--path is required for upload operations")
        if not args.path.exists():
            raise FileNotFoundError(f"Path does not exist: {args.path}")

        if args.type == 'dataset':
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
    
    else:  # download
        # Set default paths if not provided
        if not args.path:
            if args.type == 'dataset':
                local_path = Path('local_datasets') / args.repo_name.split('/')[-1]
            else:  # model
                local_path = Path('models') / args.repo_name.split('/')[-1]
        else:
            local_path = args.path
            
        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
            
        if args.type == 'dataset':
            download_dataset_from_hub(args.repo_name, local_path)
        else:  # model
            download_model_from_hub(args.repo_name, local_path)

if __name__ == "__main__":
    main()
