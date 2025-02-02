import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import tiktoken
import matplotlib.pyplot as plt
from collections import defaultdict

def load_json(file_path: str) -> List[Dict]:
    """Load JSON file and return list of entries"""
    with open(file_path, 'r') as f:
        return json.load(f)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def analyze_tokens(data: List[Dict]) -> Dict:
    """Analyze token counts for each entry"""
    stats = defaultdict(list)
    
    for entry in data:
        if entry.get('data_type') != 'training':
            continue
            
        if 'prompt' in entry and 'content' in entry['prompt']:
            prompt_tokens = count_tokens(entry['prompt']['content'])
            stats['prompt_tokens'].append(prompt_tokens)
            
        if 'chosen' in entry and 'content' in entry['chosen']:
            chosen_tokens = count_tokens(entry['chosen']['content'])
            stats['chosen_tokens'].append(chosen_tokens)
            
        if 'rejected' in entry and 'content' in entry['rejected']:
            rejected_tokens = count_tokens(entry['rejected']['content'])
            stats['rejected_tokens'].append(rejected_tokens)
            
    return stats

def generate_statistics(token_counts: Dict) -> Dict:
    """Generate statistical measures for token counts"""
    stats = {}
    
    for key, values in token_counts.items():
        if not values:
            continue
            
        stats[key] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': int(np.min(values)),
            'max': int(np.max(values)),
            'total': int(len(values))
        }
        
        # Generate histogram data
        hist, bins = np.histogram(values, bins='auto')
        stats[key]['histogram'] = {
            'counts': [int(x) for x in hist],
            'bins': [float(x) for x in bins]
        }
        
    return stats

def plot_histograms(stats: Dict, output_dir: Path):
    """Plot histograms for token distributions"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for key, data in stats.items():
        if 'histogram' not in data:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.hist(data['histogram']['bins'][:-1], 
                data['histogram']['bins'], 
                weights=data['histogram']['counts'])
        plt.title(f'Token Distribution - {key}')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / f'{key}_distribution.png')
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
    
    # Print summary
    print("\nToken Statistics Summary:")
    for key, data in stats.items():
        print(f"\n{key}:")
        for metric, value in data.items():
            if metric != 'histogram':
                print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
