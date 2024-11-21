import json
import argparse
from typing import Dict, List
import re

def clean_response(response: str) -> str:
    """
    Remove all text before '**Problem Analysis and Approach**' and add newlines.
    """
    if not response:
        return response
        
    pattern = r'\*\*Problem Analysis and Approach\*\*'
    match = re.search(pattern, response)
    
    if match:
        cleaned = response[match.start():]
        return f"\n\n{cleaned}"
    return response

def process_file(data: List[Dict]) -> List[Dict]:
    """Process all entries in the data list."""
    for entry in data:
        if 'model_response' in entry:
            entry['model_response'] = clean_response(entry['model_response'])
        if 'model_responses' in entry:
            entry['model_responses'] = [clean_response(resp) for resp in entry['model_responses']]
    return data

def main():
    parser = argparse.ArgumentParser(description='Clean model responses in JSON files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file to process')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file to save results')
    args = parser.parse_args()

    try:
        # Read input file
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Input file must contain a JSON array")
            return

        # Process the data
        cleaned_data = process_file(data)
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully processed {len(cleaned_data)} entries")
        print(f"Results saved to {args.output}")
        
    except json.JSONDecodeError:
        print(f"Error: File {args.input} is not valid JSON")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
