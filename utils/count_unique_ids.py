import json
import argparse
from collections import Counter
from typing import Dict

def count_unique_ids(filename: str) -> Dict[str, int]:
    """
    Count occurrences of each unique ID in a JSON file.
    Returns a dictionary with IDs as keys and their counts as values.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
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
    
    except json.JSONDecodeError:
        print(f"Error: {filename} is not a valid JSON file")
        return {}
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Count occurrences of unique IDs in a JSON file')
    parser.add_argument('filename', help='Path to the JSON file')
    args = parser.parse_args()
    
    id_counts = count_unique_ids(args.filename)
    
    if id_counts:
        print("\nID Occurrence Counts:")
        print("--------------------")
        for id_val, count in sorted(id_counts.items()):
            print(f"ID {id_val}: {count} occurrence{'s' if count != 1 else ''}")
        print(f"\nTotal unique IDs found: {len(id_counts)}")

if __name__ == '__main__':
    main()
