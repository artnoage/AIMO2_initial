import json
import re
from typing import Dict, List, Optional, Tuple
import argparse
from pathlib import Path

def determine_entry_type(content: str) -> str:
    """Determine if entry is full solution, analysis, or step"""
    has_step = "step" in content.lower()
    has_analysis = "analysis" in content.lower()
    
    if has_step and has_analysis:
        return "full_solution"
    elif has_analysis and not has_step:
        return "analysis"
    elif has_step and not has_analysis:
        return "step"
    else:
        return "unknown"

def validate_analysis(resp: str) -> bool:
    """Validate an analysis response"""
    if "[/INST]" in resp:
        return False
        
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 20:
        return False
        
    # Analysis should mention problem and analysis
    if "problem" not in resp.lower():
        return False
    if "analysis" not in resp.lower():
        return False
        
    return True

def validate_step(resp: str) -> bool:
    """Validate a solution step"""
    if "[/INST]" in resp:
        return False
        
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 18 or word_count > 100:
        return False
        
    # Steps should not have multiple step mentions
    step_count = resp.lower().count("step")
    return step_count <= 1

STEP_NUMBER_PATTERNS = [
    re.compile(r'^.{0,2}(\d+)[.:\)]'),
    re.compile(r'^.{0,2}\((\d+)\)'),
    re.compile(r'^.{0,2}(\d+)\s')
]

def validate_solution(solution: str) -> bool:
    """Validate a complete solution"""
    # Check for analysis section
    if "analysis" not in solution.lower():
        return False
    
    # Check analysis length
    analysis_parts = [p for p in solution.lower().split("step") if "analysis" in p.lower()]
    if analysis_parts and len(analysis_parts[0].split()) < 20:
        return False
        
    # Check for boxed answer
    if "\\boxed{" not in solution:
        return False
        
    # Split into steps and validate each
    steps = solution.lower().split("step")[1:]  # Skip text before first "step"
    if not steps:
        return False
        
    # Track step numbers found
    found_numbers = []
    
    for i, step in enumerate(steps, 1):
        # Check step length
        step_words = len(step.split())
        if step_words < 18:
            return False
        if step_words > 100:
            return False
            
        # Check step numbering
        number_found = False
        for pattern in STEP_NUMBER_PATTERNS:
            match = pattern.search(step)
            if match:
                found_numbers.append(int(match.group(1)))
                number_found = True
                break
        if not number_found:
            return False
            
    # Verify sequential step numbers
    expected_numbers = list(range(1, len(steps) + 1))
    if found_numbers != expected_numbers:
        return False
        
    return True

def validate_entry(entry: Dict) -> bool:
    """Validate a single entry from the dataset"""
    # Check if entry has required fields
    required_fields = ['prompt', 'chosen', 'rejected', 'score_chosen', 'score_rejected']
    if not all(field in entry for field in required_fields):
        return False
        
    # Get content from chosen response
    chosen_content = entry['chosen'].get('content', '')
    if not chosen_content:
        return False
        
    # Determine entry type
    entry_type = determine_entry_type(chosen_content)
    
    # Validate based on type
    if entry_type == "full_solution":
        is_valid = validate_solution(chosen_content)
    elif entry_type == "analysis":
        is_valid = validate_analysis(chosen_content)
    elif entry_type == "step":
        is_valid = validate_step(chosen_content)
    else:
        is_valid = False
        
    # Check score difference
    if is_valid:
        score_diff = abs(entry['score_chosen'] - entry['score_rejected'])
        is_valid = score_diff >= 0.2
        
    return is_valid

def main():
    parser = argparse.ArgumentParser(description='Filter hybrid data JSON file')
    parser.add_argument('input_file', type=str, help='Input JSON file path')
    parser.add_argument('output_file', type=str, help='Output JSON file path')
    args = parser.parse_args()
    
    # Read input file
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Read {len(data)} entries from {args.input_file}")
    
    # Filter entries
    valid_entries = []
    stats = {'full_solution': 0, 'analysis': 0, 'step': 0, 'unknown': 0, 'invalid': 0}
    
    for entry in data:
        if validate_entry(entry):
            valid_entries.append(entry)
            entry_type = determine_entry_type(entry['chosen']['content'])
            stats[entry_type] += 1
        else:
            stats['invalid'] += 1
    
    # Save filtered data
    with open(args.output_file, 'w') as f:
        json.dump(valid_entries, f, indent=2)
    
    # Print statistics
    print("\nValidation Results:")
    print(f"Total entries processed: {len(data)}")
    print(f"Valid entries: {len(valid_entries)}")
    print("\nValid entries by type:")
    print(f"- Full solutions: {stats['full_solution']}")
    print(f"- Analysis only: {stats['analysis']}")
    print(f"- Steps only: {stats['step']}")
    print(f"- Unknown type: {stats['unknown']}")
    print(f"\nInvalid entries: {stats['invalid']}")
    
    print(f"\nFiltered data saved to {args.output_file}")

if __name__ == "__main__":
    main()
