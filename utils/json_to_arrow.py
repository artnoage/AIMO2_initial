import json
import os
from datetime import datetime
import argparse
from datasets import Dataset, Features, Value
from typing import Dict, Any

def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of objects")
    return data

def create_arrow_dataset(data: list) -> Dataset:
    """Convert JSON data to Arrow dataset with schema"""
    
    # Infer features from first item
    if not data:
        raise ValueError("Empty dataset")
        
    sample = data[0]
    features = {}
    
    for key, value in sample.items():
        if isinstance(value, (int, bool)):
            features[key] = Value('int64')
        elif isinstance(value, float):
            features[key] = Value('float64')
        else:
            features[key] = Value('string')
    
    # Create dataset with schema
    return Dataset.from_dict(
        {k: [item.get(k) for item in data] for k in features.keys()},
        features=Features(features)
    )

def main():
    parser = argparse.ArgumentParser(description='Convert JSON to Arrow dataset')
    parser.add_argument('json_file', help='Input JSON file to convert')
    args = parser.parse_args()
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('local_dataset', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and convert data
        print(f"Loading data from {args.json_file}...")
        data = load_json_data(args.json_file)
        
        print("Converting to Arrow dataset...")
        dataset = create_arrow_dataset(data)
        
        # Save dataset
        print(f"Saving dataset to {output_dir}...")
        dataset.save_to_disk(output_dir)
        
        # Create dataset info
        dataset_info = {
            "num_examples": len(dataset),
            "features": dataset.features,
            "timestamp": timestamp,
            "source_file": args.json_file
        }
        
        # Save dataset info
        info_path = os.path.join(output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
            
        print(f"\nSuccessfully converted {len(dataset)} examples")
        print(f"Dataset saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()
