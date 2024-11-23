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

def upload_dataset_to_hub(dataset, repo_name, token):
    """Upload the dataset to HuggingFace Hub."""
    login(token)
    dataset.push_to_hub(repo_name)

def upload_model_to_hub(model_path, repo_name, token):
    """Upload a model to HuggingFace Hub."""
    login(token)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Push to hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

def main():
    parser = argparse.ArgumentParser(description='Upload datasets or models to HuggingFace Hub')
    parser.add_argument('--type', choices=['dataset', 'model'], required=True,
                      help='Type of content to upload (dataset or model)')
    parser.add_argument('--path', required=True,
                      help='Path to the dataset JSON file or model directory')
    parser.add_argument('--repo_name', required=True,
                      help='Name for the HuggingFace repository (format: username/repo-name)')
    parser.add_argument('--token', required=True,
                      help='HuggingFace API token')
    
    args = parser.parse_args()
    
    if args.type == 'dataset':
        # Handle dataset upload
        data = load_json_dataset(args.path)
        dataset = convert_to_hf_dataset(data)
        upload_dataset_to_hub(dataset, args.repo_name, args.token)
        print(f"Dataset saved locally in Arrow format at 'local_dataset'")
        print(f"Dataset successfully uploaded to {args.repo_name}")
    
    else:  # model
        # Handle model upload
        upload_model_to_hub(args.path, args.repo_name, args.token)
        print(f"Model successfully uploaded to {args.repo_name}")

if __name__ == "__main__":
    main()
