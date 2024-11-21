import argparse
import json
import random
from typing import List, Dict, Tuple, Optional
import sys

def load_json_file(filename: str) -> List[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        sys.exit(1)

def select_rejected_response(responses: List[str], verification_results: List[int], strategy: str) -> Optional[str]:
    """
    Select a rejected response based on the specified strategy.
    
    Args:
        responses: List of model responses
        verification_results: List of verification levels (0-4)
        strategy: One of 'second_best', 'random', or 'worst'
    
    Returns:
        Selected response or None if no suitable response found
    """
    # Create pairs of (score, response) for non-4 responses
    non_perfect = [(score, resp) for score, resp in zip(verification_results, responses) if score != 4]
    
    if not non_perfect:
        return None
        
    if strategy == 'second_best':
        # Sort by score descending and take highest non-4
        return max(non_perfect)[1]
    elif strategy == 'worst':
        # Take lowest score
        return min(non_perfect)[1]
    else:  # random
        # Random choice from non-4 responses
        _, selected_response = random.choice(non_perfect)
        return selected_response

def process_file(input_file: str, output_file: str, rejection_strategy: str) -> Tuple[int, int]:
    """
    Process synthetic.py output file and create DPO dataset.
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

    dpo_entries = []
    successful_pairs = 0
    
    for entry in data:
        # Find index of first verification_level 4 (if any)
        try:
            chosen_idx = entry['verification_results'].index(4)
            chosen_response = entry['model_responses'][chosen_idx]
            
            # Select rejected response based on strategy
            rejected_response = select_rejected_response(
                entry['model_responses'],
                entry['verification_results'],
                rejection_strategy
            )
            
            if rejected_response:
                dpo_entries.append({
                    "conversations": [{
                        "role": "user",
                        "content": system_prompt + "\n\n" + entry['problem']
                    }],
                    "chosen": {
                        "role": "assistant",
                        "content": chosen_response
                    },
                    "rejected": {
                        "role": "assistant",
                        "content": rejected_response
                    }
                })
                successful_pairs += 1
                
        except ValueError:
            # No verification_level 4 found, skip this entry
            continue
    
    # Save DPO entries to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_entries, f, indent=2, ensure_ascii=False)
    
    return len(data), successful_pairs

def main():
    parser = argparse.ArgumentParser(description='Create DPO dataset from synthetic.py output')
    parser.add_argument('-i', '--input', required=True,
                       help='Input JSON file from synthetic.py')
    parser.add_argument('-o', '--output', required=True,
                       help='Output JSON file for DPO dataset')
    parser.add_argument('-r', '--rejection-strategy', 
                       choices=['second_best', 'random', 'worst'],
                       default='second_best',
                       help='Strategy for selecting rejected responses')
    
    args = parser.parse_args()
    
    total_entries, successful_pairs = process_file(
        args.input, 
        args.output,
        args.rejection_strategy
    )
    
    print(f"Total entries processed: {total_entries}")
    print(f"Successfully created DPO pairs: {successful_pairs}")
    print(f"Success rate: {(successful_pairs/total_entries)*100:.2f}%")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
