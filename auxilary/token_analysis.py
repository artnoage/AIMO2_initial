import os
import sys
from typing import Dict
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def load_tokenizer(model_name: str):
    """Load just the tokenizer"""
    return AutoTokenizer.from_pretrained(model_name)

def get_dataset(dataset_name: str):
    """Load and format the dataset"""
    data = load_dataset(dataset_name)["train"]
    data = data.map(lambda x: {
        'prompt': "[INST]" + SYSTEM_PROMPT + x['problem'] + "[/INST]",
        'answer': x['answer']
    })
    return data

def analyze_token_counts(tokenizer, dataset) -> Dict:
    """Analyze token counts for all prompts in the dataset"""
    token_counts = []
    
    for example in dataset:
        tokens = tokenizer(example['prompt'])['input_ids']
        token_counts.append(len(tokens))
    
    token_counts = np.array(token_counts)
    
    stats = {
        'mean': np.mean(token_counts),
        'median': np.median(token_counts),
        'std': np.std(token_counts),
        'min': np.min(token_counts),
        'max': np.max(token_counts),
        'count': len(token_counts)
    }
    
    return stats

def main():
    # Configuration
    model_name = "mistralai/Mathstral-7B-v0.1"
    dataset_name = "Metaskepsis/Numina_medium"
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    
    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(dataset_name)
    
    # Analyze tokens
    print("Analyzing token counts...")
    stats = analyze_token_counts(tokenizer, dataset)
    
    # Print results
    print("\nToken Count Statistics:")
    print(f"Number of examples: {stats['count']}")
    print(f"Mean tokens: {stats['mean']:.2f}")
    print(f"Median tokens: {stats['median']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Min tokens: {stats['min']}")
    print(f"Max tokens: {stats['max']}")

if __name__ == "__main__":
    main()
