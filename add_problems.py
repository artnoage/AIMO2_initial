import json
import re
from typing import Dict, Optional
import argparse
from pathlib import Path

def extract_problem_from_prompt(prompt: str) -> Optional[str]:
    """Extract problem from different types of prompts"""
    
    # For FullSolutionAgent format
    full_solution_pattern = r"Solve this math problem step by step:\n\n(.*?)(?:\n\nProvide|$)"
    
    # For StepAgent format 
    step_pattern = r"Problem:\n(.*?)(?:\n\nCurrent solution:|$)"
    
    # Try full solution pattern first
    match = re.search(full_solution_pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # Try step pattern
    match = re.search(step_pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    return None

def process_json_file(input_path: str, output_path: str):
    """Process JSON file to add problem field to each entry"""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    modified = False
    
    for entry in data:
        if 'problem' not in entry and 'prompt' in entry:
            prompt_content = entry['prompt'].get('content', '')
            problem = extract_problem_from_prompt(prompt_content)
            if problem:
                entry['problem'] = problem
                modified = True
                
    if modified:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated {output_path} with problem fields")
    else:
        print(f"No changes needed for {input_path}")

def main():
    parser = argparse.ArgumentParser(description='Add missing problem fields to JSON files')
    parser.add_argument('input_files', nargs='+', help='Input JSON files to process')
    parser.add_argument('--output-dir', help='Output directory (default: same as input)', default=None)
    
    args = parser.parse_args()
    
    for input_file in args.input_files:
        input_path = Path(input_file)
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / input_path.name
        else:
            output_path = input_path.parent / f"updated_{input_path.name}"
            
        process_json_file(str(input_path), str(output_path))

if __name__ == "__main__":
    main()
