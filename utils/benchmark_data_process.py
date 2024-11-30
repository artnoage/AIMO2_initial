import json
import argparse
from typing import List, Dict
import os

def clean_json_string(text: str) -> str:
    """Clean JSON string by handling common issues like unterminated strings and missing commas"""
    # Replace any unescaped newlines inside strings
    in_string = False
    escaped = False
    cleaned = []
    i = 0
    last_token = ''
    
    while i < len(text):
        char = text[i]
        
        # Handle escape sequences
        if char == '\\' and not escaped:
            escaped = True
            cleaned.append(char)
            i += 1
            continue
            
        # Handle quotes
        if char == '"' and not escaped:
            in_string = not in_string
            
        # Clean problematic characters in strings
        if in_string:
            if char in '\n\r':
                cleaned.append(' ')
            elif char == '\t':
                cleaned.append(' ')
            elif ord(char) < 32:  # Control characters
                cleaned.append(' ')
            else:
                cleaned.append(char)
        else:
            # Handle missing commas between elements
            if char in '{[':
                if last_token and last_token not in '{[,':
                    cleaned.append(',')
                cleaned.append(char)
            elif char in '}]':
                if last_token and last_token in ',':
                    cleaned.pop()  # Remove trailing comma
                cleaned.append(char)
            elif char == '"':
                if last_token and last_token not in '{[,':
                    cleaned.append(',')
                cleaned.append(char)
            else:
                cleaned.append(char)
                
            if not char.isspace():
                last_token = char
            
        escaped = False
        i += 1
    
    # Force terminate any unterminated strings
    if in_string:
        cleaned.append('"')
        
    # Balance any unmatched braces/brackets
    result = ''.join(cleaned)
    open_braces = result.count('{')
    close_braces = result.count('}')
    open_brackets = result.count('[')
    close_brackets = result.count(']')
    
    # Add missing closing braces/brackets
    result += '}' * (open_braces - close_braces)
    result += ']' * (open_brackets - close_brackets)
    
    # Clean up any double commas that might have been introduced
    result = result.replace(',,', ',')
    
    return result

def load_benchmark_file(filename: str, clean: bool = False) -> List[Dict]:
    """
    Load benchmark results from JSON file
    Args:
        filename: Path to JSON file
        clean: Whether to clean the JSON content before parsing
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Clean the JSON content if requested
        if clean:
            content = clean_json_string(content)
            
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error at position {e.pos}, attempting cleanup...")
            # Try cleaning even if not explicitly requested
            content = clean_json_string(content)
            data = json.loads(content)
            
        # Handle both old format (with metadata) and new format (direct list)
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        return data
            
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {str(e)}\n"
                        f"Error location: line {e.lineno}, column {e.colno}")
    except Exception as e:
        raise ValueError(f"Error loading benchmark file: {str(e)}")

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
    parser.add_argument('--clean', action='store_true',
                      help='Clean JSON content before parsing (fixes formatting issues)')
    parser.add_argument('--export-cleaned', action='store_true',
                      help='Export the cleaned JSON file (only meaningful with --clean)')
    
    group = parser.add_mutually_exclusive_group()
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
        data = load_benchmark_file(args.input_file, clean=args.clean or args.clean_only)
        
        # If requested, export the cleaned data
        if args.clean and args.export_cleaned:
            base_name = os.path.splitext(args.input_file)[0]
            output_file = f"{base_name}_cleaned.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\nCleaned data saved to: {output_file}")
            return 0
            
        # For filtering operations, validate data structure
        if not isinstance(data, list):
            raise ValueError("Invalid benchmark file format: expected a list of results")
        
        # Determine operation and threshold
        if args.export_bigger is not None:
            operation, mode, threshold = 'export', 'bigger', args.export_bigger
        elif args.export_smaller is not None:
            operation, mode, threshold = 'export', 'smaller', args.export_smaller
        elif args.list_bigger is not None:
            operation, mode, threshold = 'list', 'bigger', args.list_bigger
        elif args.list_smaller is not None:
            operation, mode, threshold = 'list', 'smaller', args.list_smaller
        else:
            print("No operation specified. Use --help to see available options.")
            return 1
            
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
