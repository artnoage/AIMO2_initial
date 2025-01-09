import json
import os
from datetime import datetime
import argparse
from datasets import Dataset, Features, Value
from typing import Dict, Any, List

def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of objects")
    return data

def create_arrow_dataset(data: List[Dict[str, Any]]) -> Dataset:
    """Convert JSON data to Arrow dataset"""
    if not data:
        raise ValueError("Empty dataset")
    
    # Simply convert the data as-is
    dataset = Dataset.from_list(data)
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Convert JSON to ORPO Arrow dataset')
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
        
        print("Converting to ORPO Arrow dataset...")
        dataset = create_arrow_dataset(data)
        
        # Save dataset
        print(f"Saving dataset to {output_dir}...")
        dataset.save_to_disk(output_dir)
        
        # Create detailed dataset info
        dataset_info = {
            "num_examples": len(dataset),
            "features": {
                name: feature.dtype for name, feature in dataset.features.items()
            },
            "timestamp": timestamp,
            "source_file": args.json_file,
            "format": "ORPO",
            "version": "1.0.0",
            "score_ranges": {
                "chosen": {
                    "min": float(min(dataset['score_chosen'])),
                    "max": float(max(dataset['score_chosen'])),
                    "mean": float(sum(dataset['score_chosen']) / len(dataset))
                },
                "rejected": {
                    "min": float(min(dataset['score_rejected'])),
                    "max": float(max(dataset['score_rejected'])),
                    "mean": float(sum(dataset['score_rejected']) / len(dataset))
                }
            }
        }
        
        # Save dataset info
        info_path = os.path.join(output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
            
        print(f"\nSuccessfully converted {len(dataset)} examples")
        print(f"Dataset saved to: {output_dir}")
        print("\nScore Statistics:")
        print(f"Chosen scores: min={dataset_info['score_ranges']['chosen']['min']:.3f}, "
              f"max={dataset_info['score_ranges']['chosen']['max']:.3f}, "
              f"mean={dataset_info['score_ranges']['chosen']['mean']:.3f}")
        print(f"Rejected scores: min={dataset_info['score_ranges']['rejected']['min']:.3f}, "
              f"max={dataset_info['score_ranges']['rejected']['max']:.3f}, "
              f"mean={dataset_info['score_ranges']['rejected']['mean']:.3f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()
