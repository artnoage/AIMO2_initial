import json
import argparse
from datasets import load_dataset
from typing import Dict, List
import os

def load_json_file(filename: str) -> List[Dict]:
    """Load and parse a JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: List[Dict], filename: str):
    """Save data to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Fix missing solutions in JSON file using original dataset')
    parser.add_argument('input_file', help='JSON file with missing solutions')
    parser.add_argument('--output-file', help='Output file (defaults to input_file with _fixed suffix)')
    args = parser.parse_args()

    # Set default output filename if not provided
    if not args.output_file:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_fixed{ext}"

    # Load your JSON file
    print(f"Loading JSON file: {args.input_file}")
    json_data = load_json_file(args.input_file)

    # Load the original dataset
    print("Loading original dataset from HuggingFace...")
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    
    # Create lookup dictionary from original dataset
    solution_lookup = {str(item['id']): item['solution'] for item in dataset}

    # Counter for tracking changes
    fixed_count = 0
    missing_count = 0

    # Update solutions
    for item in json_data:
        item_id = str(item['id'])
        if item_id in solution_lookup:
            if 'solution' not in item or not item['solution']:
                item['solution'] = solution_lookup[item_id]
                fixed_count += 1
        else:
            missing_count += 1
            print(f"Warning: No solution found for ID {item_id}")

    # Save updated data
    print(f"\nSaving updated data to: {args.output_file}")
    save_json_file(json_data, args.output_file)

    # Print summary
    print(f"\nSummary:")
    print(f"Total entries processed: {len(json_data)}")
    print(f"Solutions fixed: {fixed_count}")
    print(f"IDs not found: {missing_count}")

if __name__ == "__main__":
    main()
