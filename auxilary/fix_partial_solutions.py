import json
from pathlib import Path
from typing import List, Dict
import argparse

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
    for entry in data:
        if entry.get('data_type') == 'tut_ben' and entry.get('partial_solution') is not None:
            solution = entry['solution']
            steps = solution.split('\n')
            
            # Extract wrong step from verdict
            verdict = entry.get('tutor_verdicts', [None])[0]
            if verdict and "Step" in verdict:
                try:
                    wrong_step = int(verdict.split("Step")[1].split()[0].rstrip('.:)'))
                    # Create correct partial solution including the wrong step
                    if wrong_step > 0:
                        partial_solution = "\n".join(steps[:wrong_step+1])  # Include the wrong step
                        # Add substitution if available
                        substitutions = entry.get('tutor_substitutions', [None])
                        if substitutions and substitutions[0]:
                            # Remove the wrong step and add the substitution
                            partial_solution = "\n".join(steps[:wrong_step]) + "\n" + substitutions[0]
                        entry['partial_solution'] = partial_solution
                except:
                    pass
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
