import json
import argparse
from pathlib import Path

def filter_most_common_correct(input_path: str, output_path: str = None):
    """
    Filter JSON entries keeping only those where is_most_common_correct is true.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (optional)
    """
    # Default output path if none provided
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    
    # Read and filter data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    filtered_data = [entry for entry in data if entry.get('is_most_common_correct', False)]
    
    # Save filtered data
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered {len(data)} entries to {len(filtered_data)} entries")
    print(f"Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Filter JSON entries where is_most_common_correct is true')
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    filter_most_common_correct(args.input, args.output)

if __name__ == "__main__":
    main()
