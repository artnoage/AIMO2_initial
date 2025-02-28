import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi, login
import logging
import random
import sys

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Simple print function for status updates
def log_info(message):
    print(f"[INFO] {message}")

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
    output_repo = "Metaskepsis/validation_"  # Change this to your desired repo name
    token = None  # Set your token here or use environment variable
    seed = 100
    
    # Load the first dataset: Olympiads_hard
    print(f"Loading dataset: Metaskepsis/Olympiads_hard")
    hard_dataset = load_dataset("Metaskepsis/Olympiads_hard", split="train")
    print(f"Loaded {len(hard_dataset)} examples from Metaskepsis/Olympiads_hard (train)")
    
    # Format the hard dataset
    processed_hard = hard_dataset.map(lambda x: {
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x.get('problem', x.get('question', '')),
            'id': x.get('id', str(random.randint(0, 1000000)))
        })
    
    # Shuffle and select examples from hard dataset
    processed_hard = processed_hard.shuffle(seed=seed)
    processed_hard = processed_hard.select(range(500))
    
    
    # Load the second dataset: Numina_very_hard
    print(f"Loading dataset: Metaskepsis/Numina_very_hard")
    very_hard_dataset = load_dataset("Metaskepsis/Numina_very_hard", split="train")
    print(f"Loaded {len(very_hard_dataset)} examples from Metaskepsis/Numina_very_hard (train)")
    
    # Format the very hard dataset
    processed_very_hard = very_hard_dataset.map(lambda x: {
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x.get('problem', x.get('question', '')),
            'id': x.get('id', str(random.randint(0, 1000000)))
        })
    
    # Shuffle and select examples from very hard dataset
    processed_very_hard = processed_very_hard.shuffle(seed=seed)
    processed_very_hard = processed_very_hard.select(range(500))
    
    # Load the third dataset: AI-MO/aimo-validation-aime
    print(f"Loading dataset: AI-MO/aimo-validation-aime")
    aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    print(f"Loaded {len(aime_dataset)} examples from AI-MO/aimo-validation-aime (train)")
    
    # Format the AIME dataset
    processed_aime = aime_dataset.map(lambda x: {
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x.get('problem', x.get('question', '')),
            'id': x.get('id', str(random.randint(0, 1000000)))
        })
    
    # Shuffle and select examples from AIME dataset
    processed_aime = processed_aime.shuffle(seed=seed)
    processed_aime = processed_aime.select(range(500))
    
    # Load the fourth dataset: Metaskepsis/custom100bench
    print(f"Loading dataset: Metaskepsis/custom100bench")
    custom100_dataset = load_dataset("Metaskepsis/custom100bench", split="train")
    print(f"Loaded {len(custom100_dataset)} examples from Metaskepsis/custom100bench (train)")
    
    # Format the custom100bench dataset
    processed_custom100 = custom100_dataset.map(lambda x: {
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x.get('problem', x.get('question', '')),
            'id': x.get('id', str(random.randint(0, 1000000)))
        })
    
    # Shuffle and select examples from custom100bench dataset
    processed_custom100 = processed_custom100.shuffle(seed=seed)
    processed_custom100 = processed_custom100.select(range(500))
    
    # Load the fifth dataset: Metaskepsis/custom219
    print(f"Loading dataset: Metaskepsis/custom219")
    custom219_dataset = load_dataset("Metaskepsis/custom219", split="train")
    print(f"Loaded {len(custom219_dataset)} examples from Metaskepsis/custom219 (train)")
    
    # Format the custom219 dataset
    processed_custom219 = custom219_dataset.map(lambda x: {
            'answer': x.get('answer', x.get('correct_answer', '')),
            'problem': x.get('problem', x.get('question', '')),
            'id': x.get('id', str(random.randint(0, 1000000)))
        })
    
    # Shuffle and select examples from custom219 dataset
    processed_custom219 = processed_custom219.shuffle(seed=seed)
    processed_custom219 = processed_custom219.select(range(500))


    # Concatenate the datasets
    print("Concatenating datasets")
    combined_dataset = concatenate_datasets([
        processed_hard, 
        processed_very_hard,
        processed_aime,
        processed_custom100,
        processed_custom219
    ])
    print(f"Created combined dataset with {len(combined_dataset)} examples")
    
    # Push to hub
    if output_repo != "your-username/combined-olympiads":  # Only push if repo name is changed
        hub_url = push_to_hub(combined_dataset, output_repo, token)
        print(f"Dataset available at: {hub_url}")
    else:
        print("Skipping push to hub - please set a valid output_repo name")

if __name__ == "__main__":
    main()
