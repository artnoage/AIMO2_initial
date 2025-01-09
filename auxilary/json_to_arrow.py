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
        
    # Define explicit schema for required training fields
    features = {
        'prompt': Value('string'),
        'chosen': Value('string'),
        'rejected': Value('string'),
        'score_chosen': Value('float64'),
        'score_rejected': Value('float64')
    }
    
    # Validate all entries have required fields
    for i, item in enumerate(data):
        missing = [field for field in features.keys() if field not in item]
        if missing:
            raise ValueError(f"Entry {i} is missing required fields: {missing}")
        
        # Ensure scores are numeric
        for score_field in ['score_chosen', 'score_rejected']:
            if not isinstance(item[score_field], (int, float)):
                raise ValueError(f"Entry {i}: {score_field} must be numeric, got {type(item[score_field])}")
    
    # Create dataset with schema
    return Dataset.from_dict(
        {k: [item[k] for item in data] for k in features.keys()},
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
