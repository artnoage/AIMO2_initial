import json
import argparse
import textwrap
from typing import Dict, List

def format_text(text: str, width: int = 80) -> str:
    """Format text with proper line breaks"""
    return '\n'.join(textwrap.fill(line, width=width) 
                    for line in text.split('\n'))

def format_verification_results(entry: Dict) -> str:
    """Format a single verification result entry"""
    formatted = []
    
    # Problem
    formatted.append("PROBLEM:")
    formatted.append(format_text(entry['problem']))
    formatted.append("\nMODEL RESPONSE:")
    formatted.append(format_text(entry['model_response']))
    
    if entry.get('solution'):
        formatted.append("\nSOLUTION:")
        formatted.append(format_text(entry['solution']))
    
    # Verifications
    formatted.append("\nVERIFICATIONS:")
    verifiers = entry['verifications']['verifiers']
    correctness = entry['verifications']['correctness']
    
    for verifier, is_correct in zip(verifiers, correctness):
        result = "✓" if is_correct else "✗"
        formatted.append(f"{verifier}: {result}")
    
    formatted.append("\n" + "="*80 + "\n")
    return '\n'.join(formatted)

def main():
    parser = argparse.ArgumentParser(description='Format verification results in a readable way')
    parser.add_argument('input_file', help='Input JSON file containing verification results')
    parser.add_argument('output_file', help='Output text file for formatted results')
    parser.add_argument('--width', type=int, default=80,
                       help='Maximum line width for text wrapping (default: 80)')
    
    args = parser.parse_args()
    
    try:
        # Read JSON file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format and write results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                formatted_entry = format_verification_results(entry)
                f.write(formatted_entry)
        
        print(f"Results formatted and saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
