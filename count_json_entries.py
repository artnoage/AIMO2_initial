import json
import argparse
from typing import Any, Dict, List
from pathlib import Path

def count_entries(data: Any, zero_rejected: bool = False) -> int:
    """
    Count entries in a JSON structure.
    Handles both objects and arrays.
    
    Args:
        data: JSON data structure
        zero_rejected: If True, only count entries where score_rejected is 0
    """
    if isinstance(data, list):
        if zero_rejected:
            return len([item for item in data if item.get('score_rejected', None) == 0])
        return len(data)
    elif isinstance(data, dict):
        if zero_rejected:
            return 1 if data.get('score_rejected', None) == 0 else 0
        return len(data)
    else:
        return 1

def main():
    parser = argparse.ArgumentParser(description='Count entries in a JSON file')
    parser.add_argument('file', type=str, help='Path to JSON file')
    parser.add_argument('--zero-rejected', action='store_true', 
                      help='Only count entries where score_rejected is 0')
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        count = count_entries(data, args.zero_rejected)
        print(f"Number of entries in {file_path}: {count}")
    
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
