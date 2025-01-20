import json
import argparse
from pathlib import Path
from typing import Dict, Any

def fix_alignment_spelling(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively fix 'allignment' to 'alignment' in a dictionary
    """
    if isinstance(data, dict):
        fixed_dict = {}
        for key, value in data.items():
            # Fix the key if it's 'allignment'
            new_key = 'alignment' if key == 'allignment' else key
            # Recursively fix values
            fixed_dict[new_key] = fix_alignment_spelling(value)
        return fixed_dict
    elif isinstance(data, list):
        return [fix_alignment_spelling(item) for item in data]
    else:
        return data

def main():
    parser = argparse.ArgumentParser(description='Fix alignment spelling in JSON files')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_file', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Load data
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Fix spelling
    fixed_data = fix_alignment_spelling(data)
    
    # Save fixed data
    with open(args.output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"Fixed JSON saved to {args.output_file}")

if __name__ == "__main__":
    main()
