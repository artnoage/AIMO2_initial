import json
import re
from typing import Dict, Optional, Tuple
import argparse
from pathlib import Path

# Custom JSON decoder to handle null values
def parse_null(val):
    if val == 'null':
        return None
    raise ValueError(f'Invalid null value: {val}')

# Register the custom parser
json.JSONDecoder.parse_constant = parse_null

def extract_problem_from_prompt(prompt: str) -> Tuple[Optional[str], str]:
    """Extract problem from different types of prompts and return problem and type"""
    
    # For FullSolutionAgent format
    if "Here is a mathematical problem to solve:" in prompt:
        parts = prompt.split("Here is a mathematical problem to solve:", 1)
        if len(parts) == 2:
            problem_text = parts[1].split("Please provide", 1)[0].strip()
            return problem_text, "full_solution"
    
    # For NextStepAgent format
    if "Here is a mathematical problem:" in prompt:
        parts = prompt.split("Here is a mathematical problem:", 1)
        if len(parts) == 2:
            # Extract between the header and "Your task is"
            problem_text = parts[1].split("Your task is", 1)[0].strip()
            return problem_text, "step"
            
    # For AnalysisAgent format
    if "You are a mathematical analysis expert" in prompt:
        parts = prompt.split("Here is a mathematical problem:", 1)
        if len(parts) == 2:
            problem_text = parts[1].split("Before solving", 1)[0].strip()
            return problem_text, "analysis"
    
    # Print first few chars to help debug
    preview = prompt[:100].replace('\n', '\\n')
    return None, f"unknown (preview: {preview}...)"

def process_json_file(input_path: str, output_path: str):
    """Process JSON file to add problem field to each entry"""
    
    with open(input_path, 'r') as f:
        data = json.load(f, strict=False)
        
    if not isinstance(data, list):
        print(f"Error: Input JSON must contain a list of entries")
        return
        
    modified = False
    filtered_data = []
    
    for entry in data:
        # Skip entries with null content
        if 'prompt' in entry and entry['prompt'].get('content') is None:
            continue
            
        if 'problem' in entry:
            filtered_data.append(entry)
            continue
            
        if 'prompt' in entry:
            prompt_content = entry['prompt'].get('content') or ''
            problem, _ = extract_problem_from_prompt(prompt_content)
            if problem:
                entry['problem'] = problem
                modified = True
            # Reorder fields to ensure id and problem come first
            ordered_entry = {}
            # Add id first if it exists
            if 'id' in entry:
                ordered_entry['id'] = entry['id']
            # Add problem second
            if 'problem' in entry:
                ordered_entry['problem'] = entry['problem']
            # Add remaining fields
            for key, value in entry.items():
                if key not in ['id', 'problem']:
                    ordered_entry[key] = value
            filtered_data.append(ordered_entry)
                
    if filtered_data:
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Add missing problem fields to JSON files')
    parser.add_argument('input_file', help='Input JSON file to process')
    parser.add_argument('output_file', help='Output JSON file path')
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
