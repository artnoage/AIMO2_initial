import json
import argparse
from typing import Set

def load_ids(filename: str) -> Set[str]:
    """Load IDs from a JSON file into a set"""
    with open(filename, 'r') as f:
        data = json.load(f)
        return {str(item['id']) for item in data}

def main():
    parser = argparse.ArgumentParser(description='Compare IDs between two JSON files')
    parser.add_argument('file1', help='First JSON file')
    parser.add_argument('file2', help='Second JSON file')
    args = parser.parse_args()

    try:
        ids1 = load_ids(args.file1)
        ids2 = load_ids(args.file2)

        # Compare sets
        only_in_1 = ids1 - ids2
        only_in_2 = ids2 - ids1
        common = ids1 & ids2

        print(f"\nTotal IDs in {args.file1}: {len(ids1)}")
        print(f"Total IDs in {args.file2}: {len(ids2)}")
        print(f"IDs in common: {len(common)}")
        
        if only_in_1:
            print(f"\nIDs only in {args.file1} ({len(only_in_1)}):")
            print(sorted(only_in_1))
            
        if only_in_2:
            print(f"\nIDs only in {args.file2} ({len(only_in_2)}):")
            print(sorted(only_in_2))
            
        if not (only_in_1 or only_in_2):
            print("\nThe files contain exactly the same IDs!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
