import json
import argparse
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_json(file_path: str) -> List[Dict]:
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file: {str(e)}")
        return []

def save_json(data: List[Dict], file_path: str):
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving JSON file: {str(e)}")

def extract_step_verdicts(data: List[Dict]) -> List[Dict]:
    """Extract entries that have step verdicts"""
    step_entries = []
    
    for entry in data:
        if 'messages' in entry and len(entry['messages']) >= 2:
            response = entry['messages'][1].get('content', '')
            
            # Extract verdict section
            import re
            verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
            if verdict_match:
                verdict = verdict_match.group(1).strip()
                
                # Check if verdict starts with "Step"
                if verdict.startswith("Step "):
                    step_entries.append(entry)
    
    return step_entries

def main():
    parser = argparse.ArgumentParser(description='Extract entries with step verdicts from JSON data')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('output_file', help='Path to save the filtered JSON file')
    args = parser.parse_args()
    
    # Load data
    data = load_json(args.input_file)
    if not data:
        return
    
    # Extract step verdicts
    step_entries = extract_step_verdicts(data)
    
    # Save filtered data
    save_json(step_entries, args.output_file)
    
    # Print results
    print(f"\nExtracted {len(step_entries)} entries with step verdicts")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
