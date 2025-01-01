import json
import argparse
from typing import Any, Dict, List
from pathlib import Path

def count_entries(data: Any) -> int:
    """
    Count entries in a JSON structure.
    Handles both objects and arrays.
    """
    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        return len(data)
    else:
        return 1

def main():
    parser = argparse.ArgumentParser(description='Count entries in a JSON file')
    parser.add_argument('file', type=str, help='Path to JSON file')
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        count = count_entries(data)
        print(f"Number of entries in {file_path}: {count}")
    
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
