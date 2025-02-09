import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import sys

# Add project root to path to import utils
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.benchmark_utils import NumericVerifier, extract_numeric_answer

def load_json(file_path: str) -> List[Dict]:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def create_training_pairs(entry: Dict) -> List[Dict]:
    """Create training pairs from benchmark entry if it has correct and incorrect solutions"""
    results = []
    
    # Get solutions and verify them against correct answer
    correct_solutions = []
    incorrect_solutions = []
    verifier = NumericVerifier(tolerance=1e-6)
    
    # Create list of solutions with their correctness
    for i, model_answer in enumerate(entry.get('model_answers', [])):
        if i < len(entry.get('model_solutions', [])):
            solution = entry['model_solutions'][i]
            # Skip if answer is None
            if model_answer is None:
                continue
                
            # Convert answer to numeric value
            try:
                numeric_answer, _ = extract_numeric_answer(str(model_answer))
                if numeric_answer is None:
                    continue
                    
                # Compare with correct answer
                correct_numeric, _ = extract_numeric_answer(entry['correct_answer'])
                if correct_numeric is None:
                    continue
                    
                # Check if answer is correct within tolerance
                is_correct = abs(numeric_answer - correct_numeric) <= 1e-6
                
                if is_correct:
                    correct_solutions.append(solution)
                else:
                    incorrect_solutions.append(solution)
            except Exception as e:
                print(f"Error processing answer {model_answer}: {e}")
                continue
    
    # Only proceed if we have at least one of each
    if not correct_solutions or not incorrect_solutions:
        return []
        
    # Pick first correct and incorrect solutions
    correct_sol = correct_solutions[0]
    incorrect_sol = incorrect_solutions[0]
    
    # Create light alignment (correct solution preferred)
    results.append({
        'id': entry['id'],
        'data_type': 'training',
        'example_processed_successfully': True,
        'alignment': 'light',
        'type': 'full_solution',
        'problem': entry['problem'],
        'correct_answer': entry.get('correct_answer'),
        'prompt': {'content': entry['problem'], 'role': 'user'},
        'chosen': {'content': correct_sol, 'role': 'assistant'},
        'rejected': {'content': incorrect_sol, 'role': 'assistant'},
        'score_chosen': 1.0,
        'score_rejected': 0.0
    })
    
    # Create dark alignment (incorrect solution preferred)
    results.append({
        'id': entry['id'],
        'data_type': 'training',
        'example_processed_successfully': True,
        'alignment': 'dark',
        'type': 'full_solution',
        'problem': entry['problem'],
        'correct_answer': entry.get('correct_answer'),
        'prompt': {'content': entry['problem'], 'role': 'user'},
        'chosen': {'content': incorrect_sol, 'role': 'assistant'},
        'rejected': {'content': correct_sol, 'role': 'assistant'},
        'score_chosen': 1.0,
        'score_rejected': 0.0
    })
    
    # Create judge alignment
    # Randomly decide solution positions
    correct_first = random.choice([True, False])
    judge_prompt = (
        "You are a mathematics judge. You will be presented with a problem and two proposed solutions: "
        "Solution A and Solution B. Your task is to thoroughly evaluate both solutions and determine which one "
        "demonstrates stronger reasoning and is more likely to be correct.\n\n"
        f"Problem:\n{entry['problem']}\n\n"
        f"Solution A:\n{correct_sol if correct_first else incorrect_sol}\n\n"
        f"Solution B:\n{incorrect_sol if correct_first else correct_sol}\n\n"
        "Which solution is better, A or B?"
    )
    
    results.append({
        'id': entry['id'],
        'data_type': 'training',
        'example_processed_successfully': True,
        'alignment': 'judge',
        'type': 'full_solution',
        'problem': entry['problem'],
        'correct_answer': entry.get('correct_answer'),
        'prompt': {'content': judge_prompt, 'role': 'user'},
        'chosen': {'content': 'A' if correct_first else 'B', 'role': 'assistant'},
        'rejected': {'content': 'B' if correct_first else 'A', 'role': 'assistant'},
        'score_chosen': 1.0,
        'score_rejected': 0.0
    })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Convert benchmark output to training format')
    parser.add_argument('input_file', type=str, help='Input JSON file from benchmark')
    parser.add_argument('--output-file', type=str, help='Output JSON file (default: input_file_converted.json)')
    args = parser.parse_args()
    
    # Load input data
    data = load_json(args.input_file)
    
    # Convert format
    converted_data = []
    for entry in data:
        if entry.get('data_type') == 'training':
            converted_entries = create_training_pairs(entry)
            converted_data.extend(converted_entries)
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        input_path = Path(args.input_file)
        output_path = str(input_path.parent / f"{input_path.stem}_converted{input_path.suffix}")
    
    # Save converted data
    save_json(converted_data, output_path)
    print(f"Converted {len(converted_data)} entries")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
