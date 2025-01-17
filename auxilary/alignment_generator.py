import json
from typing import Dict, List, Optional
import random

def generate_dark_and_judge(light_entry: Dict) -> List[Dict]:
    """
    Generate dark and judge entries from a light entry.
    Returns a list containing the dark and judge entries.
    """
    results = []
    
    # Extract key information
    problem = light_entry['problem']
    correct_answer = light_entry['correct_answer']
    entry_type = light_entry['type']
    
    # Create dark entry
    loki_prompt = (
        "You are a math trickster tasked with creating a deliberately incorrect but convincing solution. "
        "Your goal is to write a solution that appears mathematically sound but contains subtle errors "
        "that would fool even a careful mathematician.\n\n"
        f"Problem:\n{problem}\n\n"
        "Please provide a complete solution that:\n"
        "1. Uses correct mathematical notation and LaTeX\n"
        "2. Follows logical steps\n"
        "3. Contains subtle but significant errors\n"
        "4. Arrives at an incorrect answer\n"
        "5. Appears convincing at first glance\n\n"
        "Make sure to include analysis, step-by-step reasoning, and box the final answer using \\boxed{}"
    )
    
    dark_entry = {
        'alignment': 'dark',
        'type': entry_type,
        'problem': problem,
        'correct_answer': correct_answer,
        'prompt': {'content': loki_prompt, 'role': 'user'},
        'chosen': light_entry['rejected'],  # Flip chosen/rejected for dark
        'rejected': light_entry['chosen'],
        'score_chosen': 1.0,
        'score_rejected': 0.0
    }
    if 'id' in light_entry:
        dark_entry['id'] = light_entry['id']
    results.append(dark_entry)
    
    # Create judge entry
    # Randomly decide position of correct solution
    correct_first = random.choice([True, False])
    
    # Get solutions from light entry and remove last step from both
    def split_into_steps(solution: str) -> List[str]:
        """Split solution into steps by double newlines"""
        steps = [s.strip() for s in solution.split('\n\n') if s.strip()]
        return steps
        
    correct_steps = split_into_steps(light_entry['chosen']['content'])
    wrong_steps = split_into_steps(light_entry['rejected']['content'])
    
    # Remove last step from both solutions
    truncated_correct = "\n\n".join(correct_steps[:-1]) if len(correct_steps) > 1 else correct_steps[0]
    truncated_wrong = "\n\n".join(wrong_steps[:-1]) if len(wrong_steps) > 1 else wrong_steps[0]
    
    judge_prompt = (
        "You are a mathematics judge. You will be presented with a problem and two proposed partial or full solutions: "
        "Solution A and Solution B. Your task is to thoroughly evaluate both solutions and determine which one "
        "demonstrates stronger reasoning and is more likely to be correct.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution A:\n{truncated_correct if correct_first else truncated_wrong}\n\n"
        f"Solution B:\n{truncated_wrong if correct_first else truncated_correct}\n\n"
        "Which solution is better, A or B?"
    )
    
    judge_entry = {
        'alignment': 'judge',
        'type': entry_type,
        'problem': problem,
        'correct_answer': correct_answer,
        'prompt': {'content': judge_prompt, 'role': 'user'},
        'chosen': {'content': 'A' if correct_first else 'B', 'role': 'assistant'},
        'rejected': {'content': 'B' if correct_first else 'A', 'role': 'assistant'},
        'score_chosen': 1.0,
        'score_rejected': 0.0
    }
    if 'id' in light_entry:
        judge_entry['id'] = light_entry['id']
    results.append(judge_entry)
    
    return results

def main():
    """Process a JSON file containing light entries and generate dark/judge entries"""
    import argparse
    parser = argparse.ArgumentParser(description='Generate dark and judge entries from light entries')
    parser.add_argument('input_file', help='Input JSON file containing light entries')
    parser.add_argument('output_file', help='Output JSON file to write all entries')
    args = parser.parse_args()
    
    # Read input file
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Start with all existing entries and add alignment field
    all_entries = []
    for entry in data:
        entry_copy = entry.copy()
        entry_copy['alignment'] = 'light'
        all_entries.append(entry_copy)
    
    # Add dark and judge entries for light entries of type full_solution or recovery
    for entry in data:
        entry_with_alignment = entry.copy()
        entry_with_alignment['alignment'] = 'light'
        if entry_with_alignment['type'] in ['full_solution', 'recovery']:
            # Generate and add dark/judge entries
            all_entries.extend(generate_dark_and_judge(entry))
    
    # Write output file
    with open(args.output_file, 'w') as f:
        json.dump(all_entries, f, indent=2)
    
    print(f"Processed {len(data)} light entries")
    print(f"Generated {len(all_entries)} total entries")
    print(f"Output written to {args.output_file}")

if __name__ == "__main__":
    main()
