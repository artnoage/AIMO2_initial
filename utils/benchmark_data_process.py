import json
import argparse
from typing import List, Dict
import os

def load_benchmark_file(filename: str) -> List[Dict]:
    """Load benchmark results from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            # Handle both old format (with metadata) and new format (direct list)
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            return data
    except Exception as e:
        raise ValueError(f"Error loading benchmark file: {e}")

def calculate_success_rate(result: Dict) -> float:
    """Calculate success rate for a single result"""
    if 'attempts' not in result or 'total' not in result['attempts']:
        return 0.0
    
    total = result['attempts']['total']
    correct = result['attempts']['correct_count']
    
    return correct / total if total > 0 else 0.0

def filter_examples(results: List[Dict], threshold: float, comparison: str = 'bigger') -> List[Dict]:
    """Filter examples based on success rate threshold"""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
        
    filtered_results = []
    
    for result in results:
        success_rate = calculate_success_rate(result)
        should_include = (success_rate > threshold if comparison == 'bigger' 
                         else success_rate < threshold)
        
        if should_include:
            filtered_results.append({
                'id': result['id'],
                'problem': result['problem'],
                'correct_answer': result['correct_answer'],
                'success_rate': success_rate,
                'model_responses': result.get('model_responses', []),
                'model_answers': result.get('model_answers', []),
                'is_correct_list': result.get('is_correct_list', []),
                'attempts': result.get('attempts', {})
            })
    
    return filtered_results

def save_results(results: List[Dict], original_file: str, threshold: float, 
                operation: str, mode: str) -> None:
    """Save filtered results to a new JSON file"""
    base_name = os.path.splitext(original_file)[0]
    threshold_str = f"{int(threshold*100)}"
    output_file = f"{base_name}_{operation}_{mode}_{threshold_str}.json"
    
    # For list operation, create minimal entries with just IDs
    if operation == 'list':
        results = [{'id': result['id']} for result in results]
    
    output_data = results
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    print(f"Total examples {mode} than {threshold}: {len(results)}")

def main():
    parser = argparse.ArgumentParser(description='Process benchmark results and filter by success rate')
    parser.add_argument('input_file', help='Input benchmark JSON file')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-export-bigger', type=float,
                      help='Export full entries with success rate bigger than threshold (0-1)')
    group.add_argument('-export-smaller', type=float,
                      help='Export full entries with success rate smaller than threshold (0-1)')
    group.add_argument('-list-bigger', type=float,
                      help='List IDs with success rate bigger than threshold (0-1)')
    group.add_argument('-list-smaller', type=float,
                      help='List IDs with success rate smaller than threshold (0-1)')
    
    args = parser.parse_args()
    
    try:
        # Load and validate data
        data = load_benchmark_file(args.input_file)
        if 'results' not in data:
            raise ValueError("Invalid benchmark file format: 'results' key not found")
        
        # Determine operation and threshold
        if args.export_bigger is not None:
            operation, mode, threshold = 'export', 'bigger', args.export_bigger
        elif args.export_smaller is not None:
            operation, mode, threshold = 'export', 'smaller', args.export_smaller
        elif args.list_bigger is not None:
            operation, mode, threshold = 'list', 'bigger', args.list_bigger
        else:  # list_smaller
            operation, mode, threshold = 'list', 'smaller', args.list_smaller
            
        # Filter results
        filtered_results = filter_examples(data, threshold, mode)
        
        # Save results
        save_results(filtered_results, args.input_file, threshold, operation, mode)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
