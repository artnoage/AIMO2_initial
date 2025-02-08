import json
import argparse
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_json(file_path: str) -> List[Dict]:
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file: {str(e)}")
        return []

def analyze_verdicts(data: List[Dict]) -> Dict:
    """Analyze verdicts in the data"""
    total_entries = 0
    correct_answers = 0
    wrong_steps = 0
    wrong_approaches = 0
    invalid_verdicts = 0
    
    for entry in data:
        if 'messages' in entry and len(entry['messages']) >= 2:
            response = entry['messages'][1].get('content', '')
            
            # Extract verdict section
            import re
            verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
            if verdict_match:
                verdict = verdict_match.group(1).strip()
                total_entries += 1
                
                if verdict == "The answer is correct":
                    correct_answers += 1
                elif verdict.startswith("Step "):
                    wrong_steps += 1
                elif verdict == "The whole approach is wrong":
                    wrong_approaches += 1
                else:
                    invalid_verdicts += 1
    
    # Calculate percentages
    total = total_entries or 1  # Avoid division by zero
    stats = {
        "total_entries": total_entries,
        "correct_answers": {
            "count": correct_answers,
            "percentage": (correct_answers / total) * 100
        },
        "wrong_steps": {
            "count": wrong_steps,
            "percentage": (wrong_steps / total) * 100
        },
        "wrong_approaches": {
            "count": wrong_approaches,
            "percentage": (wrong_approaches / total) * 100
        },
        "invalid_verdicts": {
            "count": invalid_verdicts,
            "percentage": (invalid_verdicts / total) * 100
        }
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Analyze verdicts in JSON data')
    parser.add_argument('file_path', help='Path to the JSON file')
    args = parser.parse_args()
    
    # Load and analyze data
    data = load_json(args.file_path)
    if not data:
        return
    
    stats = analyze_verdicts(data)
    
    # Print results
    print("\nVerdict Analysis Results:")
    print("=" * 50)
    print(f"Total entries analyzed: {stats['total_entries']}")
    print("\nBreakdown:")
    print(f"- Correct answers: {stats['correct_answers']['count']} ({stats['correct_answers']['percentage']:.1f}%)")
    print(f"- Wrong steps: {stats['wrong_steps']['count']} ({stats['wrong_steps']['percentage']:.1f}%)")
    print(f"- Wrong approaches: {stats['wrong_approaches']['count']} ({stats['wrong_approaches']['percentage']:.1f}%)")
    print(f"- Invalid verdicts: {stats['invalid_verdicts']['count']} ({stats['invalid_verdicts']['percentage']:.1f}%)")

if __name__ == "__main__":
    main()
