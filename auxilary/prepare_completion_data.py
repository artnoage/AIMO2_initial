import json
import re
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def has_thinking_response_format(solution: str) -> bool:
    """Check if solution has proper thinking and response sections"""
    thinking_pattern = re.search(r'<thinking>(.*?)</thinking>', solution, re.DOTALL)
    response_pattern = re.search(r'<response>(.*?)</response>', solution, re.DOTALL)
    return thinking_pattern is not None and response_pattern is not None

def has_proper_step_numbering(solution: str) -> bool:
    """Check if solution has properly numbered steps in sequence"""
    # Extract response section
    response_match = re.search(r'<response>(.*?)</response>', solution, re.DOTALL)
    if not response_match:
        return False
    
    response = response_match.group(1)
    
    # Find all step markers
    step_matches = re.findall(r'<step>Step\s+(\d+):', response, re.IGNORECASE)
    if not step_matches:
        return False
    
    # Convert to integers and check sequence
    try:
        step_numbers = [int(num) for num in step_matches]
        expected_sequence = list(range(1, len(step_numbers) + 1))
        return step_numbers == expected_sequence
    except ValueError:
        return False

def extract_answer_from_solution(solution: str) -> Optional[str]:
    """Extract the boxed answer from the solution"""
    boxed_pattern = re.search(r'\\boxed{(.*?)}', solution, re.DOTALL)
    if boxed_pattern:
        return boxed_pattern.group(1).strip()
    return None

def is_numeric_match(model_answer: str, correct_answer: str, tolerance: float = 1e-6) -> bool:
    """Check if model answer numerically matches the correct answer within tolerance"""
    try:
        # Try to convert both to float
        model_num = float(model_answer.replace(',', ''))
        correct_num = float(correct_answer.replace(',', ''))
        return abs(model_num - correct_num) <= tolerance
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        return model_answer.strip() == correct_answer.strip()

def create_partial_solution(solution: str) -> Tuple[str, str]:
    """
    Create a partial solution by truncating at a random step marker.
    Returns (partial_solution, completion)
    """
    # Extract response section only
    response_match = re.search(r'<response>(.*?)</response>', solution, re.DOTALL)
    
    if not response_match:
        return solution, ""
    
    response = response_match.group(1)
    
    # Find all step markers
    step_matches = list(re.finditer(r'<step>Step\s+(\d+):', response, re.IGNORECASE))
    if len(step_matches) <= 1:
        return solution, ""  # Not enough steps to truncate
    
    # Choose a random truncation point (leave at least one step)
    truncate_at = random.randint(1, len(step_matches) - 1)
    
    # Get the position to truncate
    truncate_pos = step_matches[truncate_at].start()
    
    # Create partial solution and completion
    partial_response = response[:truncate_pos]
    remaining_response = response[truncate_pos:]
    
    # Only include the response part in the partial solution
    partial_solution = f"<response>{partial_response}"
    completion = remaining_response + "</response>"
    
    return partial_solution, completion

def process_benchmark_results(input_file: str, output_file: str, min_steps: int = 3) -> None:
    """
    Process benchmark results to create training data for completion model.
    
    Args:
        input_file: Path to input JSON file with benchmark results
        output_file: Path to output JSON file for training data
        min_steps: Minimum number of steps required in a solution
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    logger.info(f"Loaded {len(data)} entries from {input_file}")
    
    # Filter for training entries with model solutions
    training_entries = [entry for entry in data if 
                        entry.get('data_type') == 'training' and 
                        entry.get('model_solution') and
                        entry.get('is_correct')]
    
    logger.info(f"Found {len(training_entries)} training entries with correct solutions")
    
    # Filter for properly formatted solutions
    formatted_entries = []
    for entry in training_entries:
        solution = entry.get('model_solution', '')
        
        # Check format requirements
        if not has_thinking_response_format(solution):
            continue
            
        if not has_proper_step_numbering(solution):
            continue
            
        # Count steps to ensure we have enough
        step_count = len(re.findall(r'<step>Step\s+\d+:', solution, re.IGNORECASE))
        if step_count < min_steps:
            continue
            
        # Verify answer is present and correct
        model_answer = extract_answer_from_solution(solution)
        correct_answer = entry.get('correct_answer')
        if not model_answer or not correct_answer:
            continue
            
        # Double-check correctness
        if not is_numeric_match(model_answer, correct_answer):
            continue
            
        formatted_entries.append(entry)
    
    logger.info(f"Found {len(formatted_entries)} properly formatted entries with correct answers")
    
    # Create completion training data
    completion_data = []
    for entry in formatted_entries:
        solution = entry.get('model_solution', '')
        problem = entry.get('problem', '')
        correct_answer = entry.get('correct_answer', '')
        
        # Create partial solution and completion
        partial_solution, completion = create_partial_solution(solution)
        
        # Skip if we couldn't create a good partial solution
        if not completion:
            continue
            
        completion_data.append({
            'problem': problem,
            'partial_solution': partial_solution,
            'completion': completion,
            'answer': correct_answer,
            'original_id': entry.get('id')
        })
    
    logger.info(f"Created {len(completion_data)} completion training examples")
    
    # Save to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(completion_data, f, indent=2)
        logger.info(f"Saved completion training data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving output file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Process benchmark results for completion training')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to input JSON file with benchmark results')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to output JSON file for training data')
    parser.add_argument('--min-steps', type=int, default=3,
                        help='Minimum number of steps required in a solution (default: 3)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_benchmark_results(args.input, args.output, args.min_steps)

if __name__ == "__main__":
    main()
