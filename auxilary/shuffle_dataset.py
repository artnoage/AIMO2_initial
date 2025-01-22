import json
import argparse
import random
from typing import List, Dict
from pathlib import Path

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def shuffle_and_reassign_ids(data: List[Dict]) -> List[Dict]:
    """
    Shuffle the dataset and assign new sequential IDs
    
    Args:
        data: List of dictionaries containing entries
        
    Returns:
        Shuffled list with reassigned IDs
    """
    # Make a copy to avoid modifying the input
    shuffled_data = data.copy()
    
    # Shuffle the data
    random.shuffle(shuffled_data)
    
    # Assign new IDs
    for i, entry in enumerate(shuffled_data):
        entry['id'] = i
        
    return shuffled_data

def main():
    parser = argparse.ArgumentParser(description='Shuffle dataset and reassign IDs')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    try:
        # Load data
        data = load_json(args.input_file)
        print(f"Loaded {len(data)} entries")
        
        # Shuffle and reassign IDs
        shuffled_data = shuffle_and_reassign_ids(data)
        
        # Save shuffled data
        save_json(shuffled_data, args.output_file)
        print(f"Saved shuffled dataset with {len(shuffled_data)} entries")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return

if __name__ == "__main__":
    main()
