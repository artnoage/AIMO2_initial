import json
import argparse
from typing import List, Dict, Any

def filter_model_code(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter JSON entries based on model_code content.
    Keep entries that either:
    1. Don't have a model_code key
    2. Have a model_code key with content containing 'import' or 'def'
    """
    filtered_data = []
    
    for entry in data:
        # Case 1: Entry doesn't have model_code key
        if "model_code" not in entry:
            filtered_data.append(entry)
            continue
            
        # Case 2: Entry has model_code with import or def
        model_code = entry["model_code"]
        if isinstance(model_code, str) and ("import" in model_code or "def" in model_code):
            filtered_data.append(entry)
    
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description='Filter JSON entries based on model_code content')
    parser.add_argument('input_file', help='Input JSON file to filter')
    parser.add_argument('--output', '-o', default='filtered.json',
                      help='Output file name (default: filtered.json)')
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {args.input_file} is not valid JSON")
        return
    except Exception as e:
        print(f"Error reading {args.input_file}: {str(e)}")
        return
    
    if not isinstance(data, list):
        print(f"Error: {args.input_file} does not contain a JSON array")
        return
    
    # Filter the data
    filtered_data = filter_model_code(data)
    
    # Write filtered data to output file
    with open(args.output, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Original entries: {len(data)}")
    print(f"Filtered entries: {len(filtered_data)}")
    print(f"Removed entries: {len(data) - len(filtered_data)}")
    print(f"Filtered data written to {args.output}")

if __name__ == "__main__":
    main()
