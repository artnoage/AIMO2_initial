import json
import random
import re
import sympy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from latex2sympy2 import latex2sympy
from transformers import AutoTokenizer

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
    if not answer:
        return None
        
    # Check for logical operators that indicate multiple answers
    if "\\text{or}" in answer or "\\text{and}" in answer:
        return None
        
    # Clean the answer string
    clean_answer = answer.strip()
    clean_answer = re.sub(r'\\textbf{([^}]*)}', r'\1', clean_answer)  # Remove \textbf{} first   
    clean_answer = re.sub(r'\\text{[^}]*}', '', clean_answer)
    clean_answer = clean_answer.replace('\\pm', '')
    clean_answer = clean_answer.replace('\\ ', '')
    clean_answer = clean_answer.replace('\\,', '')
    clean_answer = clean_answer.replace('\\%', '')
    clean_answer = clean_answer.replace('^{\\circ}', '')  # Remove degree symbol
    clean_answer = clean_answer.replace('^\\circ', '')  # Remove degree symbol
    
    # Only split on = or \approx if there's a single term before it
    def has_single_term(text: str) -> bool:
        """Check if text has only a single term (no operators outside brackets)"""
        bracket_level = 0
        for char in text:
            if char == '{':
                bracket_level += 1
            elif char == '}':
                bracket_level -= 1
            elif bracket_level == 0 and char in '+-*/^':
                return False
        return True

    # Handle = and \approx separately
    if '=' in clean_answer:
        eq_pos = clean_answer.rfind('=')
        before_eq = clean_answer[:eq_pos].strip()
        if has_single_term(before_eq):
            clean_answer = clean_answer[eq_pos + 1:].strip()
    
    if '\\approx' in clean_answer:
        approx_pos = clean_answer.rfind('\\approx')
        before_approx = clean_answer[:approx_pos].strip()
        if has_single_term(before_approx):
            clean_answer = clean_answer[approx_pos + 8:].strip()
                
    if not clean_answer:
        return None
        
    try:
        # Parse LaTeX to sympy-compatible format
        try:
            latex_expr = latex2sympy(clean_answer)
        except:
            return None
            
        # Convert to sympy expression and evaluate
        try:
            expr = sympy.sympify(latex_expr)
        except:
            return None
            
        # Handle both single values and lists/matrices
        try:
            if hasattr(expr, 'evalf'):
                result = float(expr.evalf())
            elif isinstance(expr, list) or isinstance(expr, tuple) or (
                hasattr(expr, 'is_Matrix') and expr.is_Matrix
            ):
                return None
            else:
                result = float(expr)
            return result
        except:
            return None
    except:
        return None

def is_solution_correct(solution: str, correct_answer: str) -> bool:
    """Check if the solution's answer matches the correct answer"""
    # Extract boxed answer from solution
    boxed_pattern = re.compile(r'\\boxed\{([^}]+)\}')
    match = boxed_pattern.search(solution)
    if not match:
        return False
    solution_answer = match.group(1)
    
    # Extract numeric values
    solution_value = extract_numeric_answer(solution_answer)
    correct_value = extract_numeric_answer(correct_answer)
    
    # If either value couldn't be extracted, consider it incorrect
    if solution_value is None or correct_value is None:
        return False
        
    # Compare with small tolerance for floating point
    return abs(solution_value - correct_value) <= 1e-6

def create_tutor_prompts(data: List[Dict], max_tokens: int = 4000) -> Tuple[List[Dict], Dict]:
    """
    Process benchmark entries to create tutor prompts.
    Keeps all tags except model_solutions, creates a new prompt field.
    Filters out 90% of correct solutions to balance the dataset.
    Limits entries to those with prompts under max_tokens.
    Returns tuple of (processed_entries, statistics_dict).
    """
    processed = []
    stats = {
        'total_entries': 0,
        'skipped_no_numeric_answer': 0,
        'skipped_no_solutions': 0,
        'skipped_correct_solution': 0,
        'skipped_token_limit': 0,
        'processed': 0
    }
    
    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Metaskepsis/Skepsis_2")
    except:
        print("Warning: Failed to load Skepsis_2 tokenizer. Falling back to GPT2 tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    for entry in data:
        stats['total_entries'] += 1
        if 'data_type' not in entry or entry['data_type'] != 'training':
            continue
            
        if 'problem' not in entry or 'model_solutions' not in entry or 'correct_answer' not in entry:
            continue
            
        # Skip if we can't extract a numeric answer from the correct answer
        if extract_numeric_answer(entry['correct_answer']) is None:
            stats['skipped_no_numeric_answer'] += 1
            continue
            
        # Create new entry without model_solutions and model_answers, and change data_type
        new_entry = {k:v for k,v in entry.items() if k not in ['model_solutions', 'model_answers']}
        new_entry['data_type'] = 'tutor_prompt'
        
        # Randomly select one solution if multiple exist
        solutions = entry['model_solutions']
        if not solutions:
            stats['skipped_no_solutions'] += 1
            continue
            
        # Check each solution and filter based on correctness
        correct_solutions = []
        incorrect_solutions = []
        
        for solution in solutions:
            if is_solution_correct(solution, entry['correct_answer']):
                correct_solutions.append(solution)
            else:
                incorrect_solutions.append(solution)
        
        # Skip correct solutions with 90% probability
        if correct_solutions and random.random() < 0.9:
            if incorrect_solutions:
                selected_solution = random.choice(incorrect_solutions)
            else:
                stats['skipped_correct_solution'] += 1
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
        
        # Check token count
        prompt_tokens = len(tokenizer.encode(new_entry['prompt']))
        if prompt_tokens > max_tokens:
            stats['skipped_token_limit'] += 1
            continue
            
        processed.append(new_entry)
        stats['processed'] += 1
        
    return processed, stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create tutor prompts from benchmark results')
    parser.add_argument('input_file', help='Input JSON file from benchmark')
    parser.add_argument('output_file', help='Output JSON file for prompts')
    args = parser.parse_args()
    
    # Load data
    data = load_json(args.input_file)
    
    # Process entries
    processed, stats = create_tutor_prompts(data)
    
    # Save results
    save_json(processed, args.output_file)
    
    # Print statistics
    print("\nProcessing Statistics:")
    print(f"Total entries examined: {stats['total_entries']}")
    print(f"Entries skipped due to:")
    print(f"  - No numeric answer: {stats['skipped_no_numeric_answer']}")
    print(f"  - No solutions: {stats['skipped_no_solutions']}")
    print(f"  - Correct solution filtered: {stats['skipped_correct_solution']}")
    print(f"  - Token limit exceeded: {stats['skipped_token_limit']}")
    print(f"Final processed entries: {stats['processed']}")

if __name__ == "__main__":
    main()
