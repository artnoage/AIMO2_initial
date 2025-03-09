import os
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, load_from_disk
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
    """Convert the data to a HuggingFace Dataset and save in Arrow format with train split."""
    # First collect all possible fields across all entries
    print("Scanning dataset for all possible fields...")
    all_fields = set()
    for entry in data:
        all_fields.update(entry.keys())
    
    print(f"Found {len(all_fields)} unique fields: {', '.join(sorted(all_fields))}")
    
    # Normalize all entries to have the same fields
    print("Normalizing entries to include all fields...")
    normalized_data = []
    for entry in data:
        normalized_entry = {field: entry.get(field, None) for field in all_fields}
        normalized_data.append(normalized_entry)
    
    # Create Dataset from normalized data
    dataset = Dataset.from_list(normalized_data)
    
    # Save locally in Arrow format with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if dataset_type:
        save_path = os.path.join("local_datasets", dataset_type, timestamp)
    else:
        save_path = os.path.join("local_datasets", timestamp)
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    return dataset, save_path

def convert_to_json(arrow_path: Path, output_path: Path = None):
    """Convert an Arrow dataset to JSON format."""
    try:
        dataset = load_from_disk(str(arrow_path))
        
        # Handle both Dataset and DatasetDict formats
        if isinstance(dataset, DatasetDict):
            if 'train' not in dataset:
                raise ValueError("No train split found in dataset")
            data = dataset['train'].to_list()
        else:
            # Direct Dataset object
            data = dataset.to_list()
        
        # Ensure all entries have the same fields
        all_fields = set()
        for entry in data:
            all_fields.update(entry.keys())
        
        print(f"Found {len(all_fields)} unique fields in dataset")
        
        # Normalize all entries
        normalized_data = []
        for entry in data:
            normalized_entry = {field: entry.get(field, None) for field in all_fields}
            normalized_data.append(normalized_entry)
        
        if output_path is None:
            # Create output path based on input path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = arrow_path.parent / f"dataset_{timestamp}.json"
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=2, ensure_ascii=False)
            
        return output_path
    except Exception as e:
        raise ValueError(f"Error converting Arrow dataset: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Convert between JSON and Arrow dataset formats')
    parser.add_argument('input_path', type=Path, help='Input file/directory path')
    parser.add_argument('--type', type=str, help='Dataset type folder name (optional, for JSON to Arrow)')
    parser.add_argument('--output', type=Path, help='Output path (optional)')
    args = parser.parse_args()
    
    # Validate path exists
    if not args.input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {args.input_path}")

    try:
        # Determine conversion direction based on input path
        if args.input_path.is_file() and args.input_path.suffix == '.json':
            # JSON to Arrow conversion
            print(f"Loading JSON dataset from {args.input_path}...")
            data = load_json_dataset(args.input_path)
            print(f"Loaded {len(data)} examples from JSON")
            print("Converting to HuggingFace dataset format...")
            dataset, save_path = convert_to_hf_dataset(data, args.type)
            print(f"Dataset saved locally in Arrow format at '{save_path}'")
            print(f"Successfully converted {len(dataset)} examples with all fields normalized")
        
        elif args.input_path.is_dir():
            # Arrow to JSON conversion
            print(f"Loading Arrow dataset from {args.input_path}...")
            output_path = convert_to_json(args.input_path, args.output)
            print(f"Dataset saved as JSON at '{output_path}'")
            
        else:
            raise ValueError("Input must be either a JSON file or an Arrow dataset directory")
            
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()
