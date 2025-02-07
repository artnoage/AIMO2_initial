import json
from pathlib import Path
from typing import List, Dict
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark_utils import split_into_steps, get_partial_solutions

def load_json(file_path: str) -> List[Dict]:
    """Load JSON dataset from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def fix_partial_solutions(data: List[Dict]) -> List[Dict]:
    """Fix partial solutions by adjusting step indexing"""
    fixed_count = 0
    error_count = 0
    
    for entry in data:
        if entry.get('data_type') == 'tut_ben' and entry.get('partial_solution') is not None:
            solution = entry['solution']
            # Use proper step splitting from benchmark utils
            steps = split_into_steps(solution)
            partial_solutions = get_partial_solutions(steps)
            
            # Extract wrong step from verdict
            verdict = entry.get('tutor_verdicts', [None])[0]
            if verdict and "Step" in verdict:
                try:
                    wrong_step = int(verdict.split("Step")[1].split()[0].rstrip('.:)'))
                    print(f"\nProcessing entry with wrong step {wrong_step}")
                    print(f"Total steps found: {len(steps)}")
                    print(f"First few steps: {steps[:3]}")
                    
                    # Get partial solution up to wrong step
                    if wrong_step > 0 and wrong_step < len(partial_solutions):
                        partial_solution = partial_solutions[wrong_step - 1]  # -1 because partial_solutions is 0-based
                        # Add substitution if available
                        substitutions = entry.get('tutor_substitutions', [None])
                        if substitutions and substitutions[0]:
                            partial_solution += "\n\n" + substitutions[0]  # Add double newline for spacing
                        entry['partial_solution'] = partial_solution
                        fixed_count += 1
                    else:
                        print(f"Warning: Invalid step number {wrong_step} for solution with {len(steps)} steps")
                        error_count += 1
                except Exception as e:
                    print(f"Error processing entry: {str(e)}")
                    error_count += 1
    
    print(f"\nSummary:")
    print(f"Fixed entries: {fixed_count}")
    print(f"Errors encountered: {error_count}")
    return data

def main():
    parser = argparse.ArgumentParser(description='Fix partial solutions in dataset')
    parser.add_argument('input_path', type=str, help='Path to input JSON file')
    parser.add_argument('--output_path', type=str, help='Path to output JSON file (optional)')
    
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path or input_path.replace('.json', '_fixed.json')
    
    # Load and process data
    data = load_json(input_path)
    fixed_data = fix_partial_solutions(data)
    
    # Save results
    save_json(fixed_data, output_path)
    print(f"Processed dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
