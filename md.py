import json
import argparse
from pathlib import Path

def json_to_markdown(data: list, limit: int = 100) -> str:
    """Convert JSON entries to markdown format"""
    markdown = "# Mathematical Problems and Solutions\n\n"
    
    for i, entry in enumerate(data[:limit], 1):
        # Add problem section
        markdown += f"## Problem {i}\n\n"
        if 'problem' in entry:
            markdown += f"{entry['problem']}\n\n"
            
        # Add chosen solution if available
        if 'chosen' in entry and entry['chosen'].get('content'):
            markdown += "### Correct Solution\n\n"
            markdown += f"```\n{entry['chosen']['content']}\n```\n\n"
            
        # Add rejected solution if available
        if 'rejected' in entry and entry['rejected'].get('content'):
            markdown += "### Incorrect Solution\n\n"
            markdown += f"```\n{entry['rejected']['content']}\n```\n\n"
            
        # Add scores if available
        if 'score_chosen' in entry or 'score_rejected' in entry:
            markdown += "### Scores\n\n"
            if 'score_chosen' in entry:
                markdown += f"- Correct solution score: {entry['score_chosen']:.3f}\n"
            if 'score_rejected' in entry:
                markdown += f"- Incorrect solution score: {entry['score_rejected']:.3f}\n"
            markdown += "\n"
            
        # Add separator between problems
        markdown += "---\n\n"
        
    return markdown

def process_json_file(input_path: str, output_path: str, limit: int = 100):
    """Process JSON file and create markdown output"""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        print(f"Error: Input JSON must contain a list of entries")
        return
        
    markdown = json_to_markdown(data, limit)
    
    with open(output_path, 'w') as f:
        f.write(markdown)
    
    print(f"Created markdown file with {min(limit, len(data))} entries")

def main():
    parser = argparse.ArgumentParser(description='Convert JSON entries to markdown format')
    parser.add_argument('input_file', help='Input JSON file to process')
    parser.add_argument('output_file', help='Output markdown file path')
    parser.add_argument('--limit', type=int, default=100, 
                      help='Maximum number of entries to convert (default: 100)')
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output_file, args.limit)

if __name__ == "__main__":
    main()
