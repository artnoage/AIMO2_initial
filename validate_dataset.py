import json
import os
import sys
from typing import Dict, Any, List

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
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <path_to_json_file>")
        sys.exit(1)
        
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    
    total_entries = 0
    total_errors = 0
    
    print(f"\nChecking file: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    # Try to parse the whole file first
                    data = json.load(f)
                    
                    # Ensure data is a list of entries
                    if not isinstance(data, list):
                        data = [data]  # Convert single entry to list
                    
                    file_entries = len(data)
                    print(f"Processing {file_entries} entries in file")
                    
                    # Process each entry
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
                        
                        # Print progress for large files
                        if (i + 1) % 1000 == 0:
                            print(f"Processed {i + 1}/{file_entries} entries...")
                        
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
