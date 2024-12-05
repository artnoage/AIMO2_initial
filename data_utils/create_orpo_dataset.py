import argparse
import json
import random
from typing import List, Dict, Tuple, Optional
import sys
from transformers import AutoTokenizer
from .prompts import ANALYSIS_SYSTEM_PROMPT

def load_json_file(filename: str) -> List[Dict]:
    """Load and parse a JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_token_length(text: str, tokenizer) -> bool:
    """Check if text has at most 3000 tokens"""
    return len(tokenizer.encode(text)) <= 3000

def process_file(input_file: str, output_file: str, tokenizer) -> Tuple[int, int]:
    """Process double analysis output file and create ORPO dataset."""
    data = load_json_file(input_file)
    orpo_entries = []
    successful_pairs = 0
    

    for entry in data:
        # Only process entries with valid fields and token lengths
        if all(k in entry for k in ['problem', 'analysis_1', 'analysis_2', 'score_1', 'score_2']) and \
           check_token_length(entry['analysis_1'], tokenizer) and \
           check_token_length(entry['analysis_2'], tokenizer) and \
           entry['score_1'] != entry['score_2']:  # This condition already exists and handles your request
            
            # Always use higher score as chosen
            is_first_better = entry['score_1'] > entry['score_2']
            chosen = entry['analysis_1'] if is_first_better else entry['analysis_2']
            rejected = entry['analysis_2'] if is_first_better else entry['analysis_1']
            score_chosen = max(entry['score_1'], entry['score_2']) / 10.0
            score_rejected = min(entry['score_1'], entry['score_2']) / 10.0

            orpo_entries.append({
                "prompt": {"role": "user", "content": ANALYSIS_SYSTEM_PROMPT + "\n\n" + entry['problem']},
                "chosen": {"role": "assistant", "content": chosen},
                "rejected": {"role": "assistant", "content": rejected},
                "score_chosen": score_chosen,
                "score_rejected": score_rejected
            })
            successful_pairs += 1
    
    # Save ORPO entries to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(orpo_entries, f, indent=2, ensure_ascii=False)
    
    return len(data), successful_pairs

def main():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    parser = argparse.ArgumentParser(description='Create ORPO dataset from double analysis output')
    parser.add_argument('-i', '--input', required=True,
                       help='Input JSON file from double analysis')
    parser.add_argument('-o', '--output', required=True,
                       help='Output JSON file for ORPO dataset')
    
    args = parser.parse_args()
    
    total_entries, successful_pairs = process_file(
        args.input, 
        args.output,
        tokenizer
    )
    
    print(f"Total entries processed: {total_entries}")
    print(f"Successfully created ORPO pairs: {successful_pairs}")
    print(f"Success rate: {(successful_pairs/total_entries)*100:.2f}%")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
