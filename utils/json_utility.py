import json
import argparse
from collections import Counter
from typing import Dict, List, Optional

def clean_json_file(filename: str) -> Optional[List[Dict]]:
    """Clean JSON file by removing corrupted entries and returning valid data."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # First try standard parsing
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Standard JSON parsing failed at position {e.pos}")
                print("Attempting line-by-line parsing...")
                
                # Reset file pointer
                f.seek(0)
                data = []
                line_num = 0
                
                # Read opening bracket
                first_line = f.readline().strip()
                if first_line != '[':
                    raise ValueError("File must start with '['")
                
                # Buffer for incomplete objects
                buffer = ""
                
                for line in f:
                    line_num += 1
                    if line_num % 10000 == 0:
                        print(f"Processing line {line_num}...")
                    
                    buffer += line.strip()
                    
                    if buffer.endswith('},'):  # Complete object
                        try:
                            obj = json.loads(buffer.rstrip(','))
                            data.append(obj)
                            buffer = ""
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON at line {line_num}")
                            buffer = ""
                    elif buffer.endswith('}'):  # Last object
                        try:
                            obj = json.loads(buffer)
                            data.append(obj)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON at line {line_num}")
                
                if not data:
                    raise ValueError("No valid JSON objects found")
                
                # Write cleaned data back to file
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                print(f"Cleaned and saved file with {len(data)} valid entries")
                return data

    except Exception as e:
        print(f"Error cleaning file: {str(e)}")
        return None

def count_entries(data) -> int:
    """Count the number of entries in the data."""
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        return len(data.keys())
    else:
        raise ValueError("JSON data must contain a list or dictionary")

def count_unique_ids(data) -> Dict[str, int]:
    """
    Count occurrences of each unique ID in the data.
    Returns a dictionary with IDs as keys and their counts as values.
    """
    # Handle both list and dict JSON structures
    if isinstance(data, dict):
        items = [data]
    else:
        items = data
        
    # Count occurrences of each ID
    id_counter = Counter()
    for item in items:
        if 'id' in item:
            id_counter[str(item['id'])] += 1
            
    return dict(id_counter)

def remove_entries_by_correctness(data: List[Dict], keep_correct: bool) -> List[Dict]:
    """Remove entries based on their is_correct value"""
    if not data or not isinstance(data, list):
        return []
    
    filtered_data = [entry for entry in data if entry.get('is_correct', False) == keep_correct]
    return filtered_data

def order_by_id(data: List[Dict]) -> List[Dict]:
    """Order entries by their ID number"""
    if not data or not isinstance(data, list):
        return []
    
    # Convert ID to int for proper numerical sorting
    return sorted(data, key=lambda x: int(x.get('id', 0)))

def calculate_correct_percentage(data: List[Dict]) -> float:
    """Calculate the percentage of entries where 'is_correct' is True"""
    if not data or not isinstance(data, list):
        return 0.0
        
    total_entries = len(data)
    correct_count = sum(1 for entry in data if entry.get('is_correct', False))
    
    return (correct_count / total_entries) * 100

def main():
    parser = argparse.ArgumentParser(description='JSON file analysis utility')
    parser.add_argument('--file', '-f', required=True, help='Path to the JSON file')
    parser.add_argument('--mode', '-m', 
                      choices=['entries', 'ids', 'correct', 'remove_correct', 'remove_incorrect', 'order'],
                      required=True, 
                      help='Mode: entries, ids, correct, remove_correct, remove_incorrect, or order')
    
    args = parser.parse_args()
    
    try:
        # Clean the JSON file first
        data = clean_json_file(args.file)
        if data is None:
            print("Failed to process the JSON file")
            return
            
        if args.mode == 'entries':
            count = count_entries(data)
            print(f"Number of entries: {count}")
            
        elif args.mode == 'ids':
            id_counts = count_unique_ids(data)
            if id_counts:
                print("\nID Occurrence Counts:")
                print("--------------------")
                for id_val, count in sorted(id_counts.items()):
                    print(f"ID {id_val}: {count} occurrence{'s' if count != 1 else ''}")
                print(f"\nTotal unique IDs found: {len(id_counts)}")
            else:
                print("No IDs found in the file")
                
        elif args.mode == 'correct':
            if not isinstance(data, list):
                print("Error: JSON file must contain a list of objects for correct counting")
                return
            percentage = calculate_correct_percentage(data)
            print(f"Percentage of correct answers: {percentage:.2f}%")
            
        elif args.mode == 'remove_correct':
            filtered_data = remove_entries_by_correctness(data, keep_correct=False)
            with open(args.file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Removed correct entries. Remaining entries: {len(filtered_data)}")
            
        elif args.mode == 'remove_incorrect':
            filtered_data = remove_entries_by_correctness(data, keep_correct=True)
            with open(args.file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Removed incorrect entries. Remaining entries: {len(filtered_data)}")
            
        elif args.mode == 'order':
            ordered_data = order_by_id(data)
            with open(args.file, 'w', encoding='utf-8') as f:
                json.dump(ordered_data, f, indent=2)
            print(f"Ordered {len(ordered_data)} entries by ID number")
            
    except FileNotFoundError:
        print(f"Error: File {args.file} not found")
    except json.JSONDecodeError:
        print(f"Error: {args.file} is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
