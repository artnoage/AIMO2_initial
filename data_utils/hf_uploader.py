import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import  login
import json
import argparse

def load_json_dataset(json_path):
    """Load JSON dataset and convert it to a format suitable for HuggingFace."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_hf_dataset(data):
    """Convert the data to a HuggingFace Dataset."""
    dataset = Dataset.from_list(data)
    # Save locally in Arrow format with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"local_dataset_{timestamp}"
    dataset.save_to_disk(save_path)
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
    
    print(f"Loading model and tokenizer from {model_path}...")
    try:
        # Load model and tokenizer using standard paths
        model = AutoModelForCausalLM.from_pretrained(model_path)
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
    parser.add_argument('--path', required=True,
                      help='Path to the dataset JSON file or model directory')
    parser.add_argument('--repo_name', required=False,
                      help='Name for the HuggingFace repository (format: username/repo-name)')
    parser.add_argument('--only-data', action='store_true',
                      help='Only create local dataset without uploading to hub')
    args = parser.parse_args()
    
    if args.type == 'dataset':
        # Handle dataset
        data = load_json_dataset(args.path)
        dataset = convert_to_hf_dataset(data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Dataset saved locally in Arrow format at 'local_dataset_{timestamp}'")
        
        if not args.only_data:
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
