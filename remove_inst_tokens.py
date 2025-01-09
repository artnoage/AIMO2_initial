import json
import sys
import argparse
from typing import Dict, Any

def process_json_file(input_file: str) -> Dict[str, Any]:
    """
    Process JSON file to remove [/INST] tokens from chosen content.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Handle both single dict and list of dicts
    if isinstance(data, dict):
        data = [data]
    
    for item in data:
        if 'chosen' in item and isinstance(item['chosen'], dict):
            if 'content' in item['chosen']:
                # Remove [/INST] tokens from content
                item['chosen']['content'] = item['chosen']['content'].replace('[/INST]', '')
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Remove [/INST] tokens from chosen content in JSON files')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: overwrite input)')
    
    args = parser.parse_args()
    
    try:
        processed_data = process_json_file(args.input_file)
        
        # Determine output file
        output_file = args.output if args.output else args.input_file
        
        # Write processed data
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        print(f"Successfully processed {args.input_file}")
        if args.output:
            print(f"Output written to {args.output}")
        else:
            print("Input file updated in place")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
