import argparse
import json
from typing import List, Dict, Tuple
import sys

def load_json_file(filename: str) -> List[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        sys.exit(1)

def compare_entries(entry1: Dict, entry2: Dict) -> bool:
    """
    Compare two entries to check if they have matching problems and solutions.
    Returns True if they match, False otherwise.
    """
    return (entry1.get('problem', '') == entry2.get('problem', '') and 
            entry1.get('solution', '') == entry2.get('solution', ''))

def process_files(correct_file: str, incorrect_file: str) -> List[Dict]:
    """
    Process two JSON files and find matching IDs with same problems/solutions.
    """
    correct_data = load_json_file(correct_file)
    incorrect_data = load_json_file(incorrect_file)
    
    # Create a dictionary for faster lookup of incorrect entries
    incorrect_dict = {entry.get('id'): entry for entry in incorrect_data}
    
    matches = []
    for correct_entry in correct_data:
        correct_id = correct_entry.get('id')
        if correct_id in incorrect_dict:
            incorrect_entry = incorrect_dict[correct_id]
            if compare_entries(correct_entry, incorrect_entry):
                matches.append({
                    'id': correct_id,
                    'problem': correct_entry.get('problem'),
                    'solution': correct_entry.get('solution'),
                    'correct_response': correct_entry.get('response', ''),
                    'incorrect_response': incorrect_entry.get('response', '')
                })
    
    return matches

def main():
    parser = argparse.ArgumentParser(description='Create DPO dataset from correct and incorrect examples')
    parser.add_argument('-c', '--correct', required=True, help='JSON file with correct examples')
    parser.add_argument('-i', '--incorrect', required=True, help='JSON file with incorrect examples')
    parser.add_argument('-o', '--output', default='dpo_dataset.json', help='Output file name')
    
    args = parser.parse_args()
    
    matches = process_files(args.correct, args.incorrect)
    
    print(f"Found {len(matches)} matching entries")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
