import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

def filter_by_types(data: List[Dict], types: List[str]) -> List[Dict]:
    """
    Filter a list of dictionaries to only include entries with specified types.
    
    Args:
        data: List of dictionaries containing entries
        types: List of types to keep (e.g. ['light', 'dark', 'judge'])
        
    Returns:
        Filtered list containing only entries of specified types
    """
    return [entry for entry in data if entry.get('type') in types]

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Filter JSON data by entry types')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--types', nargs='+', required=True,
                      help='Types to keep (e.g. light dark judge)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_json(args.input_file)
    
    # Filter by types
    filtered_data = filter_by_types(data, args.types)
    
    # Save filtered data
    save_json(filtered_data, args.output_file)
    
    print(f"Filtered {len(data)} entries to {len(filtered_data)} entries")
    print(f"Kept types: {', '.join(args.types)}")

if __name__ == "__main__":
    main()
