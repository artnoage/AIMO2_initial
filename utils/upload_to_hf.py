from datasets import Dataset
from huggingface_hub import HfApi, login
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
    # Save locally in Arrow format
    dataset.save_to_disk("local_dataset")
    return dataset

def upload_to_hub(dataset, repo_name, token):
    """Upload the dataset to HuggingFace Hub."""
    # Login to Hugging Face
    login(token)
    
    # Push to hub
    dataset.push_to_hub(repo_name)

def main():
    parser = argparse.ArgumentParser(description='Convert JSON dataset to HuggingFace format and upload')
    parser.add_argument('json_path', help='Path to the JSON dataset file')
    parser.add_argument('--repo_name', default='artnoage/accept',
                        help='Name for the HuggingFace repository (format: username/dataset-name)')
    parser.add_argument('--token', default='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', 
                        help='HuggingFace API token (optional)')
    
    args = parser.parse_args()
    
    # Load and convert dataset
    data = load_json_dataset(args.json_path)
    dataset = convert_to_hf_dataset(data)
    
    # Upload to HuggingFace
    upload_to_hub(dataset, args.repo_name, args.token)
    print(f"Dataset saved locally in Arrow format at 'local_dataset'")
    print(f"Dataset successfully uploaded to {args.repo_name}")

if __name__ == "__main__":
    main()
