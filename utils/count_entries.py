import json
import argparse
from typing import Dict, List

def count_entries(filename: str) -> int:
    """Count the number of entries in a JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict):
                return len(data.keys())
            else:
                raise ValueError("JSON file must contain a list or dictionary")
    except json.JSONDecodeError:
        print(f"Error: {filename} is not a valid JSON file")
        return -1
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return -1

def main():
    parser = argparse.ArgumentParser(description='Count entries in a JSON file')
    parser.add_argument('filename', help='Path to the JSON file')
    args = parser.parse_args()
    
    count = count_entries(args.filename)
    if count >= 0:
        print(f"Number of entries: {count}")

if __name__ == '__main__':
    main()
