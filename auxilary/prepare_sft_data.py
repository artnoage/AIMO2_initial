import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def filter_for_sft(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter entries for SFT training:
    - Keep only entries where agent_type is "auxiliary"
    - Keep only entries where is_correct is true
    - Keep only entries where other_agent_correct is false
    - Keep only id, problem, model_solution (as solution), and model_answer (as answer)
    """
    filtered_data = []
    
    for entry in data:
        # Skip entries that aren't individual solutions or don't have the required fields
        if entry.get('data_type') != 'training' or 'agent_type' not in entry:
            continue
            
        # Apply our filtering criteria
        if (entry.get('agent_type') == 'auxiliary' and 
            entry.get('is_correct') == True and 
            entry.get('other_agent_correct') == False):
            
            # Create a new entry with only the fields we want
            filtered_entry = {
                'id': entry.get('id'),
                'problem': entry.get('problem'),
                'solution': entry.get('model_solution'),
                'answer': entry.get('model_answer')
            }
            
            filtered_data.append(filtered_entry)
    
    return filtered_data

def main():
    parser = argparse.ArgumentParser(description='Filter benchmark results for SFT training')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file with benchmark results')
    parser.add_argument('--output-file', type=str, default=None, 
                        help='Path to save the filtered data (default: input_file_sft.json)')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        input_path = Path(args.input_file)
        output_file = f"{input_path.stem}_sft{input_path.suffix}"
        output_path = input_path.parent / output_file
        args.output_file = str(output_path)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    
    # Filter data
    print("Filtering data for SFT...")
    filtered_data = filter_for_sft(data)
    
    # Save filtered data
    print(f"Saving {len(filtered_data)} filtered entries to {args.output_file}...")
    save_json(filtered_data, args.output_file)
    
    print("Done!")
    print(f"Original entries: {len([e for e in data if e.get('data_type') == 'training'])}")
    print(f"Filtered entries: {len(filtered_data)}")

if __name__ == "__main__":
    main()
