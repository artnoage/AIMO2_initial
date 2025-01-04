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
    full_solution_pattern = r"Solve this math problem step by step:\n\n(.*?)(?:\n\nProvide|$)"
    
    # For StepAgent format 
    step_pattern = r"Problem:\n(.*?)(?:\n\nCurrent solution:|$)"
    
    # Try full solution pattern first
    match = re.search(full_solution_pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip(), "full_solution"
        
    # Try step pattern
    match = re.search(step_pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip(), "step"
        
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
    print(f"Processing {len(data)} entries...")
    
    for i, entry in enumerate(data):
        # Skip entries with null content
        if 'prompt' in entry and entry['prompt'].get('content') is None:
            continue
            
        if 'problem' in entry:
            filtered_data.append(entry)
            continue
            
        if 'prompt' in entry:
            prompt_content = entry['prompt'].get('content') or ''
            problem, prompt_type = extract_problem_from_prompt(prompt_content)
            if problem:
                entry['problem'] = problem
                modified = True
                print(f"Entry {i}: Added problem field (type: {prompt_type})")
            else:
                print(f"Entry {i}: Could not extract problem (type: {prompt_type})")
            filtered_data.append(entry)
                
    if filtered_data:
        print(f"Writing {len(filtered_data)} entries to output file...")
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        print(f"Updated {output_path} with problem fields")
    else:
        print(f"No changes needed for {input_path}")

def main():
    parser = argparse.ArgumentParser(description='Add missing problem fields to JSON files')
    parser.add_argument('input_file', help='Input JSON file to process')
    parser.add_argument('output_file', help='Output JSON file path')
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
