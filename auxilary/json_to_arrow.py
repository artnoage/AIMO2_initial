import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from pathlib import Path
import json
import argparse

def load_json_dataset(json_path: Path):
    """Load JSON dataset and convert it to a format suitable for HuggingFace."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {json_path}")

def convert_to_hf_dataset(data, dataset_type=None):
    """Convert the data to a HuggingFace Dataset."""
    dataset = Dataset.from_list(data)
    # Save locally in Arrow format with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if dataset_type:
        save_path = os.path.join("local_datasets", dataset_type, timestamp)
    else:
        save_path = os.path.join("local_datasets", timestamp)
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    return dataset, save_path

def main():
    parser = argparse.ArgumentParser(description='Convert JSON to Arrow dataset')
    parser.add_argument('json_file', type=Path, help='Input JSON file to convert')
    parser.add_argument('--type', type=str, help='Dataset type folder name (optional)')
    args = parser.parse_args()
    
    # Validate path exists
    if not args.json_file.exists():
        raise FileNotFoundError(f"Path does not exist: {args.json_file}")

    try:
        print(f"Loading JSON dataset from {args.json_file}...")
        data = load_json_dataset(args.json_file)
        print("Converting to HuggingFace dataset format...")
        dataset, save_path = convert_to_hf_dataset(data, args.type)
        print(f"Dataset saved locally in Arrow format at '{save_path}'")
        print(f"Successfully converted {len(dataset)} examples")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()
