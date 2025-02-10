import json
import random
from pathlib import Path
from typing import Dict, List

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def create_tutor_prompts(data: List[Dict]) -> List[Dict]:
    """
    Process benchmark entries to create tutor prompts.
    Keeps all tags except model_solutions, creates a new prompt field.
    """
    processed = []
    
    for entry in data:
        if 'data_type' not in entry or entry['data_type'] != 'training':
            continue
            
        if 'problem' not in entry or 'model_solutions' not in entry:
            continue
            
        # Create new entry without model_solutions
        new_entry = {k:v for k,v in entry.items() if k != 'model_solutions'}
        
        # Randomly select one solution if multiple exist
        solutions = entry['model_solutions']
        if not solutions:
            continue
            
        selected_solution = random.choice(solutions)
        
        # Create prompt combining problem and solution
        new_entry['prompt'] = f"Here is a mathematical problem and a proposed solution:\n\nProblem:\n{entry['problem']}\n\nProposed Solution:\n{selected_solution}"
        
        processed.append(new_entry)
        
    return processed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create tutor prompts from benchmark results')
    parser.add_argument('input_file', help='Input JSON file from benchmark')
    parser.add_argument('output_file', help='Output JSON file for prompts')
    args = parser.parse_args()
    
    # Load data
    data = load_json(args.input_file)
    
    # Process entries
    processed = create_tutor_prompts(data)
    
    # Save results
    save_json(processed, args.output_file)
    print(f"Processed {len(processed)} entries")

if __name__ == "__main__":
    main()
