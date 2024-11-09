import json
import argparse
from collections import Counter
from typing import Dict, List

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

def calculate_correct_percentage(data: List[Dict]) -> float:
    """Calculate the percentage of entries where 'is_correct' is True"""
    if not data or not isinstance(data, list):
        return 0.0
        
    total_entries = len(data)
    correct_count = sum(1 for entry in data if entry.get('is_correct', False))
    
    return (correct_count / total_entries) * 100

def main():
    parser = argparse.ArgumentParser(description='JSON file analysis utility')
    parser.add_argument('filename', help='Path to the JSON file')
    parser.add_argument('--mode', '-m', choices=['entries', 'ids', 'correct'],
                      required=True, help='Count mode: entries, ids, or correct')
    
    args = parser.parse_args()
    
    try:
        with open(args.filename, 'r') as f:
            data = json.load(f)
            
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
            
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found")
    except json.JSONDecodeError:
        print(f"Error: {args.filename} is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
