import os
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import json
import argparse
from pathlib import Path

def load_json_dataset(json_path):
    """Load JSON dataset and convert it to a format suitable for HuggingFace."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_hf_dataset(data):
    """Convert the data to a HuggingFace Dataset."""
    dataset = Dataset.from_list(data)
    # Save locally in Arrow format
    dataset.save_to_disk("local_dataset")
    return dataset

def upload_dataset_to_hub(dataset, repo_name):
    """Upload the dataset to HuggingFace Hub."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token)
    dataset.push_to_hub(repo_name)

def upload_model_to_hub(model_path, repo_name):
    """Upload a model to HuggingFace Hub."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token)
    
    model_path = Path(model_path)
    print(f"Checking contents of {model_path}...")
    if model_path.is_file():
        # If a specific file is provided, use its parent directory
        model_path = model_path.parent
    
    # List all relevant files
    files = list(model_path.glob('*'))
    print("Found files:", [f.name for f in files])
    
    try:
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        print("Tokenizer loaded successfully")
        
        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)
        print("Model loaded successfully")
        
        # Push to hub
        print(f"Pushing tokenizer to {repo_name}...")
        tokenizer.push_to_hub(repo_name)
        print(f"Pushing model to {repo_name}...")
        model.push_to_hub(repo_name)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nExpected files in model directory:")
        print("- config.json")
        print("- pytorch_model.bin or model.safetensors")
        print("- tokenizer.json")
        print("- tokenizer.model (for some tokenizer types)")
        print("- special_tokens_map.json (optional)")
        print("- generation_config.json (optional)")
        raise

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Upload datasets or models to HuggingFace Hub')
    parser.add_argument('--type', choices=['dataset', 'model'], required=True,
                      help='Type of content to upload (dataset or model)')
    parser.add_argument('--path', required=True,
                      help='Path to the dataset JSON file or model directory')
    parser.add_argument('--repo_name', required=True,
                      help='Name for the HuggingFace repository (format: username/repo-name)')
    args = parser.parse_args()
    
    if args.type == 'dataset':
        # Handle dataset upload
        data = load_json_dataset(args.path)
        dataset = convert_to_hf_dataset(data)
        upload_dataset_to_hub(dataset, args.repo_name)
        print(f"Dataset saved locally in Arrow format at 'local_dataset'")
        print(f"Dataset successfully uploaded to {args.repo_name}")
    
    else:  # model
        # Handle model upload
        upload_model_to_hub(args.path, args.repo_name)
        print(f"Model successfully uploaded to {args.repo_name}")

if __name__ == "__main__":
    main()
