import json
import argparse
from typing import Dict, List

def count_correct_verdicts(file_path: str) -> Dict:
    """
    Analyze a JSON file to count tutor verdicts that are "The answer is correct"
    
    Returns:
        Dict containing total entries and count of correct verdicts
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        total_entries = 0
        correct_count = 0
        
        # Iterate through entries looking for tutor_verdicts
        for entry in data:
            if 'tutor_verdicts' in entry:
                total_entries += 1
                if "The answer is correct" in entry['tutor_verdicts']:
                    correct_count += 1
        
        return {
            "total_entries": total_entries,
            "correct_verdicts": correct_count,
            "percentage": (correct_count / total_entries * 100) if total_entries > 0 else 0
        }
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Count correct tutor verdicts in JSON file')
    parser.add_argument('file_path', help='Path to the JSON file to analyze')
    
    args = parser.parse_args()
    
    results = count_correct_verdicts(args.file_path)
    if results:
        print(f"\nResults for {args.file_path}:")
        print(f"Total entries: {results['total_entries']}")
        print(f"Correct verdicts: {results['correct_verdicts']}")
        print(f"Percentage correct: {results['percentage']:.2f}%")

if __name__ == "__main__":
    main()
