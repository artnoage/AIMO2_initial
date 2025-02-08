import json
import argparse
from typing import List, Dict
from pathlib import Path

def load_json(file_path: str) -> List[Dict]:
    """Load JSON data from file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def create_tutor_prompt(problem: str, solution: str) -> str:
    """Create the input prompt for the tutor model"""
    return (
        "Here is a mathematical problem and a proposed solution:\n\n"
        f"Problem:\n{problem}\n\n"
        f"Proposed Solution:\n{solution}\n\n"
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

def create_tutor_response(verdicts: List[str], analyses: List[str], substitutions: List[str]) -> str:
    """Create the expected output response combining tutor feedback"""
    # Take first non-null values if available
    analysis = next((a for a in analyses if a is not None), "")
    verdict = next((v for v in verdicts if v is not None), "")
    substitution = next((s for s in substitutions if s is not None), "")
    
    return (
        f"</Analysis>\n{analysis}\n<Analysis>\n\n"
        f"</Verdict>\n{verdict}\n<Verdict>\n\n"
        f"</Substitution>\n{substitution}\n<Substitution>"
    )

def create_sft_dataset(data: List[Dict]) -> List[Dict]:
    """Create SFT dataset from filtered completion benchmark results"""
    sft_data = []
    
    for entry in data:
        # Create input prompt
        prompt = create_tutor_prompt(entry['problem'], entry['solution'])
        
        # Create expected response
        response = create_tutor_response(
            entry['tutor_verdicts'],
            entry['tutor_analyses'],
            entry['tutor_substitutions']
        )
        
        # Add to dataset
        sft_data.append({
            'id': entry['id'],
            'input': prompt,
            'output': response
        })
    
    return sft_data

def main():
    parser = argparse.ArgumentParser(description='Create tutor SFT dataset from filtered completion results')
    parser.add_argument('input_file', type=str, help='Input JSON file (filtered completion results)')
    parser.add_argument('output_file', type=str, help='Output JSON file for SFT dataset')
    args = parser.parse_args()

    # Load filtered data
    data = load_json(args.input_file)
    
    # Create SFT dataset
    sft_data = create_sft_dataset(data)
    
    # Save dataset
    save_json(sft_data, args.output_file)
    
    print(f"Created SFT dataset with {len(sft_data)} examples")

if __name__ == "__main__":
    main()
