import json
import os
from typing import Dict, Any, List
import glob

def validate_entry(entry: Dict[str, Any], index: int) -> List[str]:
    """Validate a single dataset entry and return list of errors if any."""
    errors = []
    
    # Check required fields exist
    required_fields = ['prompt', 'chosen', 'rejected', 'score_chosen', 'score_rejected']
    for field in required_fields:
        if field not in entry:
            errors.append(f"Entry {index}: Missing required field '{field}'")
            continue
        
        # Check if fields are not empty
        if isinstance(entry[field], str) and not entry[field].strip():
            errors.append(f"Entry {index}: Field '{field}' is empty")
        
        # Check if score fields are numbers
        if field in ['score_chosen', 'score_rejected']:
            if not isinstance(entry[field], (int, float)):
                errors.append(f"Entry {index}: Field '{field}' must be a number, got {type(entry[field])}")
    
    return errors

def main():
    dataset_path = "/Home/stat/laschos/AIMO2_initial/local_datasets/20250109_162345"
    json_files = glob.glob(os.path.join(dataset_path, "**/*.json"), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {dataset_path}")
        return
    
    total_entries = 0
    total_errors = 0
    
    for json_file in json_files:
        print(f"\nChecking file: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    # Try to parse the whole file first
                    data = json.load(f)
                    
                    # Handle different data structures
                    if isinstance(data, list):
                        print(f"Found {len(data)} entries in file")
                        for i, entry in enumerate(data):
                            if not isinstance(entry, dict):
                                print(f"Entry {i} is not a dictionary: {type(entry)}")
                                total_errors += 1
                                continue
                            errors = validate_entry(entry, i)
                            if errors:
                                print(f"\nErrors in entry {i}:")
                                for error in errors:
                                    print(f"  - {error}")
                                total_errors += len(errors)
                            total_entries += 1
                    elif isinstance(data, dict):
                        print("Found 1 entry in file (single dictionary)")
                        errors = validate_entry(data, 0)
                        if errors:
                            print("\nErrors in entry:")
                            for error in errors:
                                print(f"  - {error}")
                            total_errors += len(errors)
                        total_entries += 1
                    else:
                        print(f"Unexpected data type in {json_file}: {type(data)}")
                        total_errors += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON in {json_file}: {str(e)}")
                    total_errors += 1
                    
        except Exception as e:
            print(f"Error reading file {json_file}: {str(e)}")
            total_errors += 1
    
    print(f"\nValidation complete!")
    print(f"Total entries checked: {total_entries}")
    print(f"Total errors found: {total_errors}")

if __name__ == "__main__":
    main()
