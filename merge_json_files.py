import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]  # Wrap single objects in a list
            return data
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def merge_json_files(input_dir: str, output_file: str) -> None:
    """Merge all JSON files from input directory into a single output file."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return

    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files")
    
    # Merge all data, keeping only specified fields
    merged_data = []
    for file_path in json_files:
        print(f"Processing {file_path.name}...")
        data = load_json_file(file_path)
        for entry in data:
            if all(field in entry for field in ["id", "problem","prompt", "chosen", "rejected", "score_chosen", "score_rejected"]):
                filtered_entry = {
                    "id": entry["id"],
                    "problem": entry["problem"],
                    "prompt": entry["prompt"],
                    "chosen": entry["chosen"],
                    "rejected": entry["rejected"],
                    "score_chosen": entry["score_chosen"],
                    "score_rejected": entry["score_rejected"],
                    "rejected_part_solution": entry["rejected_part_solution"]

                }
                merged_data.append(filtered_entry)

    # Save merged data
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"\nSuccessfully merged {len(json_files)} files into {output_file}")
        print(f"Total entries: {len(merged_data)}")
    except Exception as e:
        print(f"Error writing output file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Merge multiple JSON files into one')
    parser.add_argument('input_dir', help='Directory containing JSON files')
    parser.add_argument('output_file', help='Output JSON file path')
    args = parser.parse_args()

    merge_json_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
