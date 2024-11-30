import json
import argparse
from typing import List, Dict
import os

def load_benchmark_file(filename: str) -> Dict:
    """Load benchmark results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading benchmark file: {e}")

def calculate_success_rate(result: Dict) -> float:
    """Calculate success rate for a single result"""
    if 'attempts' not in result or 'total' not in result['attempts']:
        return 0.0
    
    total = result['attempts']['total']
    correct = result['attempts']['correct_count']
    
    return correct / total if total > 0 else 0.0

def filter_successful_examples(data: Dict, threshold: float) -> List[Dict]:
    """Filter examples with success rate above threshold"""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
        
    filtered_results = []
    
    for result in data['results']:
        success_rate = calculate_success_rate(result)
        if success_rate > threshold:
            filtered_results.append({
                'id': result['id'],
                'problem': result['problem'],
                'correct_answer': result['correct_answer'],
                'success_rate': success_rate
            })
    
    return filtered_results

def save_filtered_results(results: List[Dict], original_file: str, threshold: float) -> None:
    """Save filtered results to a new JSON file"""
    base_name = os.path.splitext(original_file)[0]
    output_file = f"{base_name}_filtered_{int(threshold*100)}.json"
    
    output = {
        'metadata': {
            'original_file': original_file,
            'threshold': threshold,
            'total_examples': len(results)
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFiltered results saved to: {output_file}")
    print(f"Total examples meeting threshold: {len(results)}")

def main():
    parser = argparse.ArgumentParser(description='Process benchmark results and filter by success rate')
    parser.add_argument('input_file', help='Input benchmark JSON file')
    parser.add_argument('-export', type=float, required=True,
                      help='Success rate threshold (0-1) for filtering examples')
    
    args = parser.parse_args()
    
    try:
        # Load and validate data
        data = load_benchmark_file(args.input_file)
        if 'results' not in data:
            raise ValueError("Invalid benchmark file format: 'results' key not found")
            
        # Filter results
        filtered_results = filter_successful_examples(data, args.export)
        
        # Save filtered results
        save_filtered_results(filtered_results, args.input_file, args.export)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
