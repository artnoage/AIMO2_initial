import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import defaultdict

def load_json(file_path: str) -> List[Dict]:
    """Load JSON file and return list of entries"""
    with open(file_path, 'r') as f:
        return json.load(f)

def count_tokens(text: str, model: str = "Metaskepsis/Skepsis_2") -> int:
    """Count tokens in text using HuggingFace tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return len(tokenizer.encode(text))
    except Exception as e:
        print(f"Warning: Failed to load {model} tokenizer. Falling back to GPT2 tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return len(tokenizer.encode(text))

def analyze_tokens(data: List[Dict]) -> Dict:
    """Analyze token counts for each entry by type (ORPO or SFT)"""
    stats = {
        'orpo': {
            'light': defaultdict(list),
            'dark': defaultdict(list),
            'judge': defaultdict(list)
        },
        'sft': defaultdict(list),
        'grpo': defaultdict(list)
    }
    
    for entry in data:
        data_type = entry.get('data_type')
        if data_type not in ['training', 'tutor_prompt']:
            continue
            
        # Handle tutor prompts (from create_tutor_prompts)
        if data_type == 'tutor_prompt':
            if 'prompt' in entry:
                prompt_tokens = count_tokens(entry['prompt'])
                stats['grpo']['prompt_tokens'].append(prompt_tokens)
            if 'model_solution' in entry:
                solution_tokens = count_tokens(entry['model_solution'])
                stats['grpo']['solution_tokens'].append(solution_tokens)
            continue
            
        # Check if entry is SFT format (from tutor_generator)
        if 'messages' in entry:
            # SFT format has messages array with user/assistant pairs
            for i, msg in enumerate(entry['messages']):
                if msg['role'] == 'user':
                    prompt_tokens = count_tokens(msg['content'])
                    stats['sft']['prompt_tokens'].append(prompt_tokens)
                elif msg['role'] == 'assistant':
                    completion_tokens = count_tokens(msg['content'])
                    stats['sft']['completion_tokens'].append(completion_tokens)
        else:
            # ORPO format
            alignment = entry.get('alignment', 'judge')
            
            if 'prompt' in entry and 'content' in entry['prompt']:
                prompt_tokens = count_tokens(entry['prompt']['content'])
                stats['orpo'][alignment]['prompt_tokens'].append(prompt_tokens)
                
            if 'chosen' in entry and 'content' in entry['chosen']:
                chosen_tokens = count_tokens(entry['chosen']['content'])
                stats['orpo'][alignment]['chosen_tokens'].append(chosen_tokens)
                
            if 'rejected' in entry and 'content' in entry['rejected']:
                rejected_tokens = count_tokens(entry['rejected']['content'])
                stats['orpo'][alignment]['rejected_tokens'].append(rejected_tokens)
            
    return stats

def generate_statistics(token_counts: Dict) -> Dict:
    """Generate statistical measures for token counts"""
    stats = {}
    
    for alignment, alignment_data in token_counts.items():
        stats[alignment] = {}
        for key, values in alignment_data.items():
            if not values:
                continue
                
            stats[alignment][key] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': int(np.min(values)),
                'max': int(np.max(values)),
                'total': int(len(values))
            }
            
            # Generate histogram data
            hist, bins = np.histogram(values, bins='auto')
            stats[alignment][key]['histogram'] = {
                'counts': [int(x) for x in hist],
                'bins': [float(x) for x in bins]
            }
    
    return stats

def plot_histograms(stats: Dict, output_dir: Path):
    """Plot histograms for token distributions by type and alignment"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle ORPO data
    if 'orpo' in stats:
        orpo_dir = output_dir / 'orpo'
        orpo_dir.mkdir(exist_ok=True)
        
        for alignment, alignment_stats in stats['orpo'].items():
            alignment_dir = orpo_dir / alignment
            alignment_dir.mkdir(exist_ok=True)
            
            for key, data in alignment_stats.items():
                if 'histogram' not in data:
                    continue
                    
                plt.figure(figsize=(10, 6))
                plt.hist(data['histogram']['bins'][:-1], 
                        data['histogram']['bins'], 
                        weights=data['histogram']['counts'])
                plt.title(f'ORPO Token Distribution - {alignment} - {key}')
                plt.xlabel('Number of Tokens')
                plt.ylabel('Frequency')
                plt.savefig(alignment_dir / f'{key}_distribution.png')
                plt.close()
    
    # Handle SFT data
    if 'sft' in stats:
        sft_dir = output_dir / 'sft'
        sft_dir.mkdir(exist_ok=True)
        
        for key, data in stats['sft'].items():
            if 'histogram' not in data:
                continue
                
            plt.figure(figsize=(10, 6))
            plt.hist(data['histogram']['bins'][:-1], 
                    data['histogram']['bins'], 
                    weights=data['histogram']['counts'])
            plt.title(f'SFT Token Distribution - {key}')
            plt.xlabel('Number of Tokens')
            plt.ylabel('Frequency')
            plt.savefig(sft_dir / f'{key}_distribution.png')
            plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze token counts in JSON dataset')
    parser.add_argument('input_file', type=str, help='Path to input JSON file')
    parser.add_argument('--output-dir', type=str, default='token_stats',
                      help='Directory for output files')
    args = parser.parse_args()
    
    # Load and analyze data
    data = load_json(args.input_file)
    token_counts = analyze_tokens(data)
    stats = generate_statistics(token_counts)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    with open(output_dir / 'token_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate plots
    plot_histograms(stats, output_dir)
    
    # Print summary by alignment
    print("\nToken Statistics Summary:")
    for alignment, alignment_stats in stats.items():
        print(f"\n=== {alignment.upper()} ===")
        for key, data in alignment_stats.items():
            print(f"\n{key}:")
            for metric, value in data.items():
                if metric != 'histogram':
                    print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
