import os
import sys
from typing import Dict, List, Tuple
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

def analyze_token_counts(tokenizer, dataset) -> Tuple[Dict, List[int]]:
    """Analyze token counts for all prompts in the dataset"""
    token_counts = []
    indices_under_800 = []
    
    for i, example in enumerate(dataset):
        tokens = tokenizer(example['prompt'])['input_ids']
        count = len(tokens)
        token_counts.append(count)
        if count <= 800:
            indices_under_800.append(i)
    
    token_counts = np.array(token_counts)
    
    stats = {
        'mean': np.mean(token_counts),
        'median': np.median(token_counts),
        'std': np.std(token_counts),
        'min': np.min(token_counts),
        'max': np.max(token_counts),
        'count': len(token_counts),
        'above_800': np.sum(token_counts > 800),
        'above_1000': np.sum(token_counts > 1000)
    }
    
    return stats, indices_under_800

def process_dataset(dataset_name: str, tokenizer) -> None:
    """Process a single dataset, analyze tokens and save filtered version"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(dataset_name)
    
    # Analyze tokens
    print("Analyzing token counts...")
    stats, indices_under_800 = analyze_token_counts(tokenizer, dataset)
    
    # Print results
    print("\nToken Count Statistics:")
    print(f"Number of examples: {stats['count']}")
    print(f"Mean tokens: {stats['mean']:.2f}")
    print(f"Median tokens: {stats['median']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Min tokens: {stats['min']}")
    print(f"Max tokens: {stats['max']}")
    print(f"Examples with >800 tokens: {stats['above_800']} ({(stats['above_800']/stats['count']*100):.1f}%)")
    print(f"Examples with >1000 tokens: {stats['above_1000']} ({(stats['above_1000']/stats['count']*100):.1f}%)")
    
    # Create filtered dataset
    filtered_dataset = dataset.select(indices_under_800)
    filtered_name = f"{dataset_name}_filtered"
    
    # Save to Hub
    print(f"\nSaving filtered dataset ({len(filtered_dataset)} examples) to {filtered_name}")
    filtered_dataset.push_to_hub(filtered_name)

def main():
    # Configuration
    model_name = "mistralai/Mathstral-7B-v0.1"
    datasets = [
        "Metaskepsis/Numina_medium",
        "Metaskepsis/Numina_hard",
        "Metaskepsis/Numina_very_hard"
    ]
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)
    
    # Process each dataset
    for dataset_name in datasets:
        process_dataset(dataset_name, tokenizer)

if __name__ == "__main__":
    main()
