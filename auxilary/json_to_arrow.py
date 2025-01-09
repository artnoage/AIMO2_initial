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
    """Convert JSON data to Arrow dataset with automatic schema inference"""
    
    if not data:
        raise ValueError("Empty dataset")

    # Automatically infer types from the first item
    first_item = data[0]
    features = {}
    
    def infer_type(value):
        if isinstance(value, bool):
            return Value('bool')
        elif isinstance(value, int):
            return Value('int64')
        elif isinstance(value, float):
            return Value('float64')
        elif isinstance(value, str):
            return Value('string')
        elif isinstance(value, list):
            return Value('string') if all(isinstance(x, str) for x in value) else Value('string')
        else:
            return Value('string')
    
    # Build schema from all fields in the data
    for item in data:
        for key, value in item.items():
            if key not in features:
                features[key] = infer_type(value)
    
    # Check if this might be ORPO training data and validate if so
    orpo_fields = {'prompt', 'chosen', 'rejected', 'score_chosen', 'score_rejected'}
    if orpo_fields.issubset(features.keys()):
        print("Detected ORPO training data format, validating fields...")
        for i, item in enumerate(data):
            # Ensure scores are numeric
            for score_field in ['score_chosen', 'score_rejected']:
                if not isinstance(item[score_field], (int, float)):
                    raise ValueError(f"Entry {i}: {score_field} must be numeric for ORPO training, got {type(item[score_field])}")
    
    # Create dataset with inferred schema
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
    output_dir = os.path.join('local_datasets', timestamp)
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
