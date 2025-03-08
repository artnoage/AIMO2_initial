import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi, login
import logging
import random
import sys
import re
import signal
from contextlib import contextmanager

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Simple print function for status updates
def log_info(message):
    print(f"[INFO] {message}")

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def extract_numeric_answer(answer: str, debug: bool = False) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value from a LaTeX answer string.
    First tries to evaluate using sympy, then falls back to direct float conversion.
    Returns float if found, None otherwise.
    """
    if not answer:
        return None, "No answer provided" if debug else None
        
    # Check for logical operators that indicate multiple answers
    if "\\text{or}" in answer or "\\text{and}" in answer:
        return None, "Answer contains 'or'/'and' operators" if debug else None
        
    # Clean the answer string
    clean_answer = answer.strip()
    clean_answer = re.sub(r'\\textbf{([^}]*)}', r'\1', clean_answer)  # Remove \textbf{} first   
    clean_answer = re.sub(r'\\text{[^}]*}', '', clean_answer)
    clean_answer = clean_answer.replace('\\pm', '')
    clean_answer = clean_answer.replace('\\ ', '')
    clean_answer = clean_answer.replace('\\,', '')
    clean_answer = clean_answer.replace('\\%', '')
    clean_answer = clean_answer.replace('^{\\circ}', '')  # Remove degree symbol
    clean_answer = clean_answer.replace('^\\circ', '')  # Remove degree symbol
    
    # Only split on = or \approx if there's a single term before it
    def has_single_term(text: str) -> bool:
        """Check if text has only a single term (no operators outside brackets)"""
        bracket_level = 0
        for char in text:
            if char == '{':
                bracket_level += 1
            elif char == '}':
                bracket_level -= 1
            elif bracket_level == 0 and char in '+-*/^':
                return False
        return True

    # Handle = and \approx separately
    if '=' in clean_answer:
        eq_pos = clean_answer.rfind('=')
        before_eq = clean_answer[:eq_pos].strip()
        if has_single_term(before_eq):
            clean_answer = clean_answer[eq_pos + 1:].strip()
    
    if '\\approx' in clean_answer:
        approx_pos = clean_answer.rfind('\\approx')
        before_approx = clean_answer[:approx_pos].strip()
        if has_single_term(before_approx):
            clean_answer = clean_answer[approx_pos + 8:].strip()
                
    if not clean_answer:
        return None, "Empty answer after cleaning" if debug else None
    
    # Simple numeric check first
    try:
        result = float(clean_answer)
        return result, f"Direct conversion: {clean_answer} -> {result}" if debug else None
    except ValueError:
        pass
    
    # If not a simple number, try more complex parsing
    try:
        # For now, just return None as we don't have latex2sympy imported
        # In a real implementation, you would use latex2sympy here
        return None, f"Complex LaTeX parsing not implemented for: {clean_answer}" if debug else None
    except Exception as e:
        return None, f"Error: {str(e)} on input: {clean_answer}" if debug else None

SYSTEM_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.\n\n
    <thinking>
    First, analyze the problem in depth and outline your approach.\n 
    This section should capture your reasoning, including any abstract thoughts or potential strategies.\n  
    Feel free to refine or correct your ideas as you work toward the solution.\n  
    </thinking>
    <response>\n
    <step>Step 1: Begin with the first calculation or operation\n
    Show your work clearly using LaTeX notation</step>\n\n
    <step>Step 2: Continue with the next logical step\n
    Each step should be numbered and self-contained</step>\n\n
    <step>Step N: In your final step, state your conclusion\n
    Put your final answer in \\boxed{}</step>\n
    </response>\n\n"""





def push_to_hub(dataset: Dataset, repo_name: str, token: Optional[str] = None) -> str:
    """
    Push a dataset to the Hugging Face Hub.
    
    Args:
        dataset: The dataset to push
        repo_name: The repository name on Hugging Face
        token: Hugging Face API token (optional if already logged in)
        
    Returns:
        URL of the pushed dataset
    """
    try:
        # Login if token is provided
        if token:
            login(token)
            print("Logged in to Hugging Face Hub")
        
        # Push the dataset
        print(f"Pushing dataset to {repo_name}")
        dataset.push_to_hub(repo_name)
        print(f"Successfully pushed dataset to {repo_name}")
        
        return f"https://huggingface.co/datasets/{repo_name}"
    except Exception as e:
        print(f"ERROR: Failed to push dataset to hub: {str(e)}")
        raise

def main():
    # Set fixed parameters
    output_repo = "Metaskepsis/validation_set"  # Change this to your desired repo name
    token = None  # Set your token here or use environment variable
    seed = 100
    
    # Load the first dataset: Olympiads_hard
    print(f"Loading dataset: Metaskepsis/Olympiads_hard")
    hard_dataset = load_dataset("Metaskepsis/Olympiads_hard", split="train")
    print(f"Loaded {len(hard_dataset)} examples from Metaskepsis/Olympiads_hard (train)")
    
    # Format the hard dataset - ensure answer is string type
    processed_hard = hard_dataset.map(lambda x: {
            'answer': str(x.get('answer', x.get('correct_answer', ''))),
            'problem': x.get('problem', x.get('question', ''))
        })
    
    # Shuffle and select examples from hard dataset
    processed_hard = processed_hard.shuffle(seed=seed)
    processed_hard = processed_hard.select(range(500))
    
    
    # Load the second dataset: Numina_very_hard
    print(f"Loading dataset: Metaskepsis/Numina_very_hard")
    very_hard_dataset = load_dataset("Metaskepsis/Numina_very_hard", split="train")
    print(f"Loaded {len(very_hard_dataset)} examples from Metaskepsis/Numina_very_hard (train)")
    
    # Format the very hard dataset - ensure answer is string type
    processed_very_hard = very_hard_dataset.map(lambda x: {
            'answer': str(x.get('answer', x.get('correct_answer', ''))),
            'problem': x.get('problem', x.get('question', ''))
        })
    
    # Shuffle and select examples from very hard dataset
    processed_very_hard = processed_very_hard.shuffle(seed=seed)
    processed_very_hard = processed_very_hard.select(range(250))
    
    # Load the third dataset: AI-MO/aimo-validation-aime
    print(f"Loading dataset: AI-MO/aimo-validation-aime")
    aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    print(f"Loaded {len(aime_dataset)} examples from AI-MO/aimo-validation-aime (train)")
    
    # Format the AIME dataset - ensure answer is string type
    processed_aime = aime_dataset.map(lambda x: {
            'answer': str(x.get('answer', x.get('correct_answer', ''))),
            'problem': x.get('problem', x.get('question', ''))
        })
    
    # Shuffle and select examples from AIME dataset
    processed_aime = processed_aime.shuffle(seed=seed)
    processed_aime = processed_aime.select(range(min(500, len(processed_aime))))

    
    # Load the fourth dataset: Metaskepsis/custom100bench
    print(f"Loading dataset: Metaskepsis/custom100bench")
    custom100_dataset = load_dataset("Metaskepsis/custom100bench", split="train")
    print(f"Loaded {len(custom100_dataset)} examples from Metaskepsis/custom100bench (train)")
    
    # Format the custom100bench dataset - ensure answer is string type
    processed_custom100 = custom100_dataset.map(lambda x: {
            'answer': str(x.get('answer', x.get('correct_answer', ''))),
            'problem': x.get('problem', x.get('question', ''))
        })
    
    # Shuffle and select examples from custom100bench dataset
    processed_custom100 = processed_custom100.shuffle(seed=seed)
    processed_custom100 = processed_custom100.select(range(min(500, len(processed_custom100))))
 
    
    # Load the fifth dataset: Metaskepsis/custom219
    print(f"Loading dataset: Metaskepsis/custom219")
    custom219_dataset = load_dataset("Metaskepsis/custom219", split="train")
    print(f"Loaded {len(custom219_dataset)} examples from Metaskepsis/custom219 (train)")
    
    # Format the custom219 dataset - ensure answer is string type
    processed_custom219 = custom219_dataset.map(lambda x: {
            'answer': str(x.get('answer', x.get('correct_answer', ''))),
            'problem': x.get('problem', x.get('question', ''))
        })
    
    # Shuffle and select examples from custom219 dataset
    processed_custom219 = processed_custom219.shuffle(seed=seed)
    processed_custom219 = processed_custom219.select(range(min(500, len(processed_custom219))))



    # Create a new dataset from scratch with consistent types
    print("Creating a new dataset with consistent schema")
    
    # Collect all examples with the right schema
    all_examples = []
    
    # Helper function to extract examples with consistent schema and filter answers
    def extract_examples(dataset):
        examples = []
        for item in dataset:
            answer = str(item['answer'])
            
            # Count occurrences of "="
            equals_count = answer.count("=")
            
            # Skip entries with 2 or more "=" occurrences
            if equals_count >= 2:
                continue
                
            # If there's exactly one "=", keep only what's on the right
            if equals_count == 1:
                answer = answer.split("=")[1].strip()
            
            # Check if the answer has a valid numeric value
            numeric_value, _ = extract_numeric_answer(answer)
            if numeric_value is None:
                continue
                
            examples.append({
                'answer': answer,
                'problem': str(item['problem']),
                'numeric_value': numeric_value
            })
        return examples
    
    # Extract examples from each dataset
    all_examples.extend(extract_examples(processed_hard))
    all_examples.extend(extract_examples(processed_very_hard))
    all_examples.extend(extract_examples(processed_aime))
    all_examples.extend(extract_examples(processed_custom100))
    all_examples.extend(extract_examples(processed_custom219))
    
    # Create a new dataset from the collected examples
    print(f"Creating dataset from {len(all_examples)} collected examples")
    combined_dataset = Dataset.from_list(all_examples)
    
    # Add new IDs to the combined dataset (just the index number)
    def add_ids(example, idx):
        example['id'] = f"{idx}"
        return example
    
    combined_dataset = combined_dataset.map(add_ids, with_indices=True)
    print(f"Created combined dataset with {len(combined_dataset)} examples, each with a unique ID")
    
    # Push to hub
    if output_repo != "your-username/combined-olympiads":  # Only push if repo name is changed
        hub_url = push_to_hub(combined_dataset, output_repo, token)
        print(f"Dataset available at: {hub_url}")
    else:
        print("Skipping push to hub - please set a valid output_repo name")

if __name__ == "__main__":
    main()
