import json
import argparse
from datasets import load_dataset
from typing import Dict, List
import os
from huggingface_hub import HfApi

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
    username = HfApi().whoami()["name"]
    dataset = load_dataset(f"{username}/Numina-Olympiads", split="train")
    
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
import json
import argparse
from typing import Dict, List, Tuple
import os

def load_json_file(filename: str) -> List[Dict]:
    """Load and parse a JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_entries(entry1: Dict, entry2: Dict) -> Tuple[bool, bool]:
    """
    Compare two entries and return tuple of booleans:
    (problem_matches, solution_matches)
    """
    problem_matches = entry1.get('problem', '') == entry2.get('problem', '')
    solution_matches = entry1.get('solution', '') == entry2.get('solution', '')
    return problem_matches, solution_matches

def main():
    parser = argparse.ArgumentParser(description='Create DPO dataset from two JSON files')
    parser.add_argument('-c', '--correct', required=True,
                      help='JSON file with correct examples')
    parser.add_argument('-i', '--incorrect', required=True,
                      help='JSON file with incorrect examples')
    args = parser.parse_args()

    # Load both JSON files
    print(f"Loading correct examples from: {args.correct}")
    correct_data = load_json_file(args.correct)
    
    print(f"Loading incorrect examples from: {args.incorrect}")
    incorrect_data = load_json_file(args.incorrect)

    # Create lookup dictionary for incorrect data
    incorrect_lookup = {str(item['id']): item for item in incorrect_data}

    # Statistics
    total_processed = 0
    matching_ids = 0
    matching_problems = 0
    matching_solutions = 0

    # Compare entries
    for correct_entry in correct_data:
        total_processed += 1
        correct_id = str(correct_entry['id'])
        
        if correct_id in incorrect_lookup:
            matching_ids += 1
            incorrect_entry = incorrect_lookup[correct_id]
            
            problem_matches, solution_matches = compare_entries(
                correct_entry, incorrect_entry)
            
            if problem_matches:
                matching_problems += 1
            if solution_matches:
                matching_solutions += 1
                
            if problem_matches and not solution_matches:
                print(f"Found potential DPO pair for ID {correct_id}:")
                print(f"  Problem matches: {problem_matches}")
                print(f"  Solution matches: {solution_matches}")

    # Print summary
    print(f"\nSummary:")
    print(f"Total entries processed: {total_processed}")
    print(f"Matching IDs found: {matching_ids}")
    print(f"Matching problems: {matching_problems}")
    print(f"Matching solutions: {matching_solutions}")
    print(f"Potential DPO pairs: {matching_problems - matching_solutions}")

if __name__ == "__main__":
    main()
