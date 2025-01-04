import json
import argparse
from pathlib import Path

def clean_problem_text(problem: str) -> str:
    """Clean problem text by removing everything after 'Before solving'"""
    if "Before solving" in problem:
        problem = problem.split("Before solving")[0].strip()
    return problem

def process_json_file(input_path: str, output_path: str):
    """Process JSON file to clean problem fields"""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        print(f"Error: Input JSON must contain a list of entries")
        return
        
    cleaned_data = []
    
    for entry in data:
        if 'problem' in entry:
            # Clean the problem text
            entry['problem'] = clean_problem_text(entry['problem'])
            cleaned_data.append(entry)
            
    if cleaned_data:
        with open(output_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print(f"Processed {len(cleaned_data)} entries")

def main():
    parser = argparse.ArgumentParser(description='Clean problem fields in JSON files')
    parser.add_argument('input_file', help='Input JSON file to process')
    parser.add_argument('output_file', help='Output JSON file path')
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
