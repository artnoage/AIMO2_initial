import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def extract_numeric_answer(answer: str) -> Optional[float]:
    """Extract numeric value from a LaTeX answer string"""
    try:
        # Remove LaTeX formatting and convert to float
        clean_answer = answer.replace('\\boxed{', '').replace('}', '').strip()
        return float(clean_answer)
    except (ValueError, AttributeError):
        return None

def is_solution_correct(solution: str, correct_answer: str) -> bool:
    """Check if the solution's answer matches the correct answer"""
    # Extract numeric values
    solution_value = extract_numeric_answer(solution)
    correct_value = extract_numeric_answer(correct_answer)
    
    # If either value couldn't be extracted, consider it incorrect
    if solution_value is None or correct_value is None:
        return False
        
    # Compare with small tolerance for floating point
    return abs(solution_value - correct_value) <= 1e-6

def create_tutor_prompts(data: List[Dict]) -> List[Dict]:
    """
    Process benchmark entries to create tutor prompts.
    Keeps all tags except model_solutions, creates a new prompt field.
    Filters out 90% of correct solutions to balance the dataset.
    """
    processed = []
    
    for entry in data:
        if 'data_type' not in entry or entry['data_type'] != 'training':
            continue
            
        if 'problem' not in entry or 'model_solutions' not in entry:
            continue
            
        # Create new entry without model_solutions and model_answers, and change data_type
        new_entry = {k:v for k,v in entry.items() if k not in ['model_solutions', 'model_answers']}
        new_entry['data_type'] = 'tutor_prompt'
        
        # Randomly select one solution if multiple exist
        solutions = entry['model_solutions']
        if not solutions:
            continue
            
        # Check each solution and filter based on correctness
        correct_solutions = []
        incorrect_solutions = []
        
        for solution in solutions:
            if is_solution_correct(solution, entry['answer']):
                correct_solutions.append(solution)
            else:
                incorrect_solutions.append(solution)
        
        # Skip correct solutions with 90% probability
        if correct_solutions and random.random() < 0.9:
            if incorrect_solutions:
                selected_solution = random.choice(incorrect_solutions)
            else:
                continue  # Skip if no incorrect solutions available
        else:
            # Use any solution (correct or incorrect)
            selected_solution = random.choice(solutions)
            
        new_entry['model_solution'] = selected_solution
        
        # Create prompt with tutor structure
        new_entry['prompt'] = (
            f"Here is a mathematical problem and a proposed solution:\n\n"
            f"Problem:\n{entry['problem']}\n\n"
            f"Proposed Solution:\n{selected_solution}\n\n"
            "Please analyze this solution and:\n"
            "1. Provide a brief analysis of the solution approach\n"
            "2. Carefully examine each step from the beginning and identify the VERY FIRST point where the logic goes wrong\n"
            "3. If there's a wrong step, suggest how to correct it\n\n"
            "Format your response exactly as:\n\n"
            "</Analysis>\n"
            "Analyze the solution approach and reasoning here\n"
            "<Analysis>\n\n"
            "</Verdict>\n"
            "Either: 'Step X' (where X is the FIRST step number where the logic becomes incorrect)\n"
            "Or: 'The whole approach is wrong' (if the approach is fundamentally flawed from the start)\n"
            "Or: 'The answer is correct' (if no errors are found)\n"
            "<Verdict>\n\n"
            "</Substitution>\n"
            "If a specific step is wrong, write 'Step X: ' followed by the correct version of that step\n"
            "Otherwise leave this section empty\n"
            "<Substitution>"
        )
        
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
