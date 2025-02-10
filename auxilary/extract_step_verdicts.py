import json
import argparse
import random
import re
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

def save_json(data: List[Dict], file_path: str):
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving JSON file: {str(e)}")

def extract_verdicts(data: List[Dict], correct_ratio: float = 0.25, 
                    wrong_steps_ratio: float = 1.0, 
                    wrong_approach_ratio: float = 1.0) -> List[Dict]:
    """
    Extract entries with different ratios for each verdict type
    
    Args:
        data: List of data entries
        correct_ratio: Ratio of correct answers to keep (0.0 to 1.0)
        wrong_steps_ratio: Ratio of wrong step entries to keep (0.0 to 1.0)
        wrong_approach_ratio: Ratio of wrong approach entries to keep (0.0 to 1.0)
    """
    correct_entries = []
    wrong_steps_entries = []
    wrong_approach_entries = []
    
    for entry in data:
        if 'messages' in entry and len(entry['messages']) >= 2:
            response = entry['messages'][1].get('content', '')
            
            # Extract verdict section
            verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
            if verdict_match:
                verdict = verdict_match.group(1).strip()
                
                # Remove any extra whitespace and quotes
                verdict = verdict.strip('"').strip()
                
                if verdict == "The answer is correct":
                    correct_entries.append(entry)
                elif verdict.startswith("Step "):
                    wrong_steps_entries.append(entry)
                elif verdict == "The whole approach is wrong":
                    wrong_approach_entries.append(entry)
    
    # Sample entries according to ratios
    sampled_correct = random.sample(correct_entries, 
                                  k=int(len(correct_entries) * correct_ratio))
    sampled_wrong_steps = random.sample(wrong_steps_entries, 
                                      k=int(len(wrong_steps_entries) * wrong_steps_ratio))
    sampled_wrong_approach = random.sample(wrong_approach_entries, 
                                         k=int(len(wrong_approach_entries) * wrong_approach_ratio))
    
    # Combine all sampled entries
    result = sampled_correct + sampled_wrong_steps + sampled_wrong_approach
    random.shuffle(result)  # Shuffle to mix different types
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract entries with different verdict ratios')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('output_file', help='Path to save the filtered JSON file')
    parser.add_argument('--correct-ratio', type=float, default=0.9,
                      help='Ratio of correct answers to keep (default: 0.25)')
    parser.add_argument('--wrong-steps-ratio', type=float, default=1.0,
                      help='Ratio of wrong step entries to keep (default: 1.0)')
    parser.add_argument('--wrong-approach-ratio', type=float, default=1.0,
                      help='Ratio of wrong approach entries to keep (default: 1.0)')
    args = parser.parse_args()
    
    # Load data
    data = load_json(args.input_file)
    if not data:
        return
    
    # Extract verdicts with specified ratios
    filtered_entries = extract_verdicts(
        data,
        correct_ratio=args.correct_ratio,
        wrong_steps_ratio=args.wrong_steps_ratio,
        wrong_approach_ratio=args.wrong_approach_ratio
    )
    
    # Save filtered data
    save_json(filtered_entries, args.output_file)
    
    # Count entries by category
    category_counts = {"correct_answers": 0, "wrong_steps": 0, "wrong_approaches": 0}
    
    for entry in filtered_entries:
        response = entry['messages'][1].get('content', '')
        verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
        if verdict_match:
            verdict = verdict_match.group(1).strip().strip('"').strip()
            
            if verdict == "The answer is correct":
                category_counts["correct_answers"] += 1
            elif verdict.startswith("Step "):
                category_counts["wrong_steps"] += 1
            elif verdict == "The whole approach is wrong":
                category_counts["wrong_approaches"] += 1
    
    # Print results
    print(f"\nExtracted entries by category:")
    print(f"- Correct answers: {category_counts['correct_answers']}")
    print(f"- Wrong steps: {category_counts['wrong_steps']}")
    print(f"- Wrong approaches: {category_counts['wrong_approaches']}")
    print(f"Total entries: {len(filtered_entries)}")
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()
