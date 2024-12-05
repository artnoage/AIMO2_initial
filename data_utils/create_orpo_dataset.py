import argparse
import json
import random
from typing import List, Dict, Tuple, Optional
import sys
from transformers import AutoTokenizer

def load_json_file(filename: str) -> List[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        sys.exit(1)

def check_token_length(text: str, tokenizer) -> bool:
    """Check if text has at most 3000 tokens"""
    return len(tokenizer.encode(text)) <= 3000

def process_file(input_file: str, output_file: str, tokenizer) -> Tuple[int, int]:
    """
    Process double analysis output file and create ORPO dataset.
    Returns: (total_entries, successful_pairs)
    """
    data = load_json_file(input_file)
    
    system_prompt = """You are a precise mathematical problem solver. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification. 
▪ Use LaTeX notation

FORMAT:

**Problem Analysis and Approach**:
1. Start by categorizing the problem (e.g., "This is an inequality problem involving algebraic identities" or "This is a combinatorial proof").
2. List specific tools or theorems that will guide your solution (e.g., "AM-GM inequality," "Basic algebraic manipulations").

**PROOF**:
Example format for each step:
Given: \\( a, b, c > 0 \\) and \\( a + b + c = 3 \\). Prove that \\( abc \\leq 1 \\).

Step 1. By the AM-GM inequality, \\( \\frac{a + b + c}{3} \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Apply AM-GM inequality to \\( a, b, c \\)]  
Step 2. Substituting \\( a + b + c = 3 \\), we get \\( 1 \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Replace with given sum condition]  
Step 3. Cube both sides to eliminate the root: \\( 1 \\geq abc \\) \\hspace{10pt} [Cube both sides to solve for \\( abc \\)]  
Step 4. Thus, \\( abc \\leq 1 \\), as required.  

For each step, clearly state the action, use concise LaTeX notation, and provide a justification in brackets.

**ANSWER**:
\\(\\boxed{\\text{result}}\\) """

    orpo_entries = []
    successful_pairs = 0
    
    for entry in data:
        try:
            # Skip entries without required fields
            if not all(k in entry for k in ['problem', 'analysis_1', 'analysis_2', 'score_1', 'score_2']):
                continue
                
            # Skip if either analysis is too long
            if not (check_token_length(entry['analysis_1'], tokenizer) and 
                   check_token_length(entry['analysis_2'], tokenizer)):
                continue

            # Determine chosen and rejected based on scores
            if entry['score_1'] > entry['score_2']:
                chosen = entry['analysis_1']
                rejected = entry['analysis_2']
                score_chosen = entry['score_1'] / 10.0  # Normalize to 0-1 range
                score_rejected = entry['score_2'] / 10.0
            elif entry['score_2'] > entry['score_1']:
                chosen = entry['analysis_2']
                rejected = entry['analysis_1']
                score_chosen = entry['score_2'] / 10.0
                score_rejected = entry['score_1'] / 10.0
            else:
                # Skip if scores are equal
                continue

            orpo_entries.append({
                "prompt": {"role": "user", "content": system_prompt + "\n\n" + entry['problem']},
                "chosen": {"role": "assistant", "content": chosen},
                "rejected": {"role": "assistant", "content": rejected},
                "score_chosen": score_chosen,
                "score_rejected": score_rejected
            })
            successful_pairs += 1
                
        except Exception as e:
            print(f"Error processing entry: {str(e)}")
            continue
    
    # Save ORPO entries to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(orpo_entries, f, indent=2, ensure_ascii=False)
    
    return len(data), successful_pairs

def main():
    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    except Exception as e:
        print(f"Error initializing tokenizer: {str(e)}")
        sys.exit(1)

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
