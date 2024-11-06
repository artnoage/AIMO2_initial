import json
import argparse
from typing import Dict, List

def calculate_correct_percentage(data: List[Dict]) -> float:
    """
    Calculate the percentage of entries where 'is_correct' is True
    """
    if not data:
        return 0.0
        
    total_entries = len(data)
    correct_count = sum(1 for entry in data if entry.get('is_correct', False))
    
    return (correct_count / total_entries) * 100

def main():
    parser = argparse.ArgumentParser(description='Calculate percentage of correct answers in JSON file')
    parser.add_argument('filename', help='Path to the JSON file')
    
    args = parser.parse_args()
    
    try:
        with open(args.filename, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: JSON file must contain a list of objects")
            return
            
        percentage = calculate_correct_percentage(data)
        print(f"Percentage of correct answers: {percentage:.2f}%")
        
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
