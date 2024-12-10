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

def detect_dataset_type(path: Path) -> str:
    """Detect if path points to a JSON file or Arrow dataset directory."""
    if path.is_file() and path.suffix.lower() == '.json':
        return 'json'
    elif path.is_dir() and any(f.name == 'dataset_info.json' for f in path.glob('*')):
        return 'arrow'
    else:
        raise ValueError(
            f"Invalid dataset path: {path}\n"
            "Path must be either:\n"
            "- A .json file containing dataset\n"
            "- A directory containing an Arrow dataset"
        )

def load_json_dataset(json_path: Path):
    """Load JSON dataset and convert it to a format suitable for HuggingFace."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {json_path}")

def convert_to_hf_dataset(data):
    """Convert the data to a HuggingFace Dataset."""
    dataset = Dataset.from_list(data)
    # Save locally in Arrow format with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join("local_datasets", timestamp)
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    return dataset

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
            repo_name,
            callbacks=[lambda x: pbar.update(x.percentage)]
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
    
    parser = argparse.ArgumentParser(description='Upload datasets or models to HuggingFace Hub')
    parser.add_argument('--type', choices=['dataset', 'model'], required=True,
                      help='Type of content to upload (dataset or model)')
    parser.add_argument('--path', required=True, type=Path,
                      help='Path to the dataset JSON file or model directory')
    parser.add_argument('--repo_name', required=False,
                      help='Name for the HuggingFace repository (format: username/repo-name)')
    parser.add_argument('--only-data', action='store_true',
                      help='Only create local dataset without uploading to hub')
    parser.add_argument('--upload-only', action='store_true',
                      help='Upload an existing Arrow dataset from disk')
    args = parser.parse_args()
    
    if args.type == 'dataset':
        # Validate path exists
        if not args.path.exists():
            raise FileNotFoundError(f"Path does not exist: {args.path}")

        try:
            # Detect and validate dataset type
            dataset_type = detect_dataset_type(args.path)
            
            if args.upload_only and dataset_type != 'arrow':
                raise ValueError("--upload-only requires an Arrow dataset directory")
            elif not args.upload_only and dataset_type != 'json':
                raise ValueError("JSON file required when not using --upload-only")

            # Handle dataset
            if args.upload_only:
                # Load existing Arrow dataset
                print(f"Loading Arrow dataset from {args.path}...")
                dataset = load_from_disk(str(args.path))
                print("Dataset loaded successfully")
            else:
                # Process JSON to Arrow dataset
                print(f"Loading JSON dataset from {args.path}...")
                data = load_json_dataset(args.path)
                print("Converting to HuggingFace dataset format...")
                dataset = convert_to_hf_dataset(data)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                print(f"Dataset saved locally in Arrow format at 'local_datasets/{timestamp}'")
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            raise
            
        if not args.only_data or args.upload_only:
            if not args.repo_name:
                raise ValueError("--repo_name is required when not using --only-data")
            upload_dataset_to_hub(dataset, args.repo_name)
            print(f"Dataset successfully uploaded to {args.repo_name}")
    
    else:  # model
        # Handle model upload
        upload_model_to_hub(args.path, args.repo_name)
        print(f"Model successfully uploaded to {args.repo_name}")

if __name__ == "__main__":
    main()
