import os
import sys
import argparse
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_from_disk, Dataset
import markdown


# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_preparation import prepare_combined_data
from utils.agents import (
    FULLSOLUTION_SYSTEM_PROMPT, 
    PROGRAMMER_SYSTEM_PROMPT, 
    FINALIZATION_SYSTEM_PROMPT, 
    TUTOR_SYSTEM_PROMPT
)

print(FULLSOLUTION_SYSTEM_PROMPT)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dataset_preview')

def clean_prompt_for_display(prompt: str) -> str:
    """
    Display prompt text verbatim in markdown
    Only replace escaped newlines with actual newlines for readability
    """
    # Replace escaped newlines with actual newlines
    prompt = prompt.replace('\\n', '\n')
    return prompt

def format_example_as_markdown(example: Dict, example_type: str, index: int) -> str:
    """Format a dataset example as markdown for preview"""
    md = f"## Example {index}: {example_type.capitalize()}\n\n"
    
    # Add prompt (cleaned for display)
    md += "### Prompt\n\n"
    md += "```\n"
    md += clean_prompt_for_display(example['prompt'])
    md += "\n```\n\n"
    
    # Add example-specific fields
    if example_type == 'finalization':
        md += "### Partial Solution\n\n"
        md += f"```\n{example['partial_solution']}\n```\n\n"
        
        md += "### Full Solution\n\n"
        md += f"```\n{example['full_solution']}\n```\n\n"
    
    if example_type == 'tutor':
        md += "### Full Solution to Evaluate\n\n"
        md += f"```\n{example['full_solution']}\n```\n\n"
        
        md += "### Is Correct\n\n"
        md += f"{example['is_correct']}\n\n"
        
        if example.get('wrong_step') is not None:
            md += "### Wrong Step\n\n"
            md += f"{example['wrong_step']}\n\n"
    
    # Add answer for all types
    md += "### Expected Answer\n\n"
    md += f"{example['answer']}\n\n"
    
    # Add horizontal rule for separation
    md += "---\n\n"
    
    return md

def create_dataset_preview(dataset_path: str, output_path: str, num_examples: int = 5):
    """
    Create a markdown preview of a dataset with equal distribution of example types.
    
    Args:
        dataset_path: Path to the dataset
        output_path: Path to save the markdown preview
        num_examples: Number of examples of each type to include
    """
    logger.info(f"Loading dataset from {dataset_path}")
    data = load_from_disk(dataset_path)
    
    # Create equal distribution of example types
    distribution = {
        'solution': 0.25,
        'programming': 0.25,
        'finalization': 0.25,
        'tutor': 0.25
    }
    
    logger.info("Preparing combined dataset with equal distribution")
    combined_data = prepare_combined_data(
        data,
        FULLSOLUTION_SYSTEM_PROMPT,
        FINALIZATION_SYSTEM_PROMPT,
        PROGRAMMER_SYSTEM_PROMPT,
        TUTOR_SYSTEM_PROMPT,
        distribution=distribution
    )
    
    # Group examples by type
    examples_by_type = {}
    for example in combined_data:
        example_type = example.get('example_type', 'unknown')
        if example_type not in examples_by_type:
            examples_by_type[example_type] = []
        
        # Check if example has all required fields
        try:
            # Basic validation to ensure we can process this example
            if example_type == 'finalization' and not example.get('partial_solution'):
                logger.warning(f"Skipping finalization example without partial_solution")
                continue
                
            if example_type == 'tutor' and not example.get('full_solution'):
                logger.warning(f"Skipping tutor example without full_solution")
                continue
                
            # Add the example if it passes validation
            examples_by_type[example_type].append(example)
        except Exception as e:
            logger.warning(f"Error processing example: {str(e)}")
            continue
    
    # Create markdown content
    md_content = "# Dataset Preview\n\n"
    md_content += f"Dataset path: `{dataset_path}`\n\n"
    md_content += "## Distribution\n\n"
    
    # Add distribution statistics
    for example_type, examples in examples_by_type.items():
        md_content += f"- **{example_type.capitalize()}**: {len(examples)} examples ({len(examples)/len(combined_data)*100:.1f}%)\n"
    
    md_content += "\n\n"
    
    # Add examples of each type
    for example_type, examples in examples_by_type.items():
        md_content += f"# {example_type.capitalize()} Examples\n\n"
        
        # Randomly select examples
        selected_examples = random.sample(
            examples, 
            min(num_examples, len(examples))
        )
        
        # Format each example
        for i, example in enumerate(selected_examples):
            md_content += format_example_as_markdown(example, example_type, i+1)
    
    # Write markdown to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Dataset preview saved to {output_path}")
    
    # Also create an HTML version for easier viewing
    html_path = output_path.replace('.md', '.html')
    html_content = markdown.markdown(md_content)
    
    # Add some basic styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Dataset Preview</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            code {{ font-family: monospace; }}
            h1, h2, h3 {{ color: #333; }}
            hr {{ border: 0; border-top: 1px solid #ddd; margin: 30px 0; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    
    logger.info(f"HTML preview saved to {html_path}")
    
    return combined_data

def main():
    parser = argparse.ArgumentParser(description="Create a preview of a dataset with equal distribution of example types")
    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument("--output", "-o", default="dataset_preview.md", help="Path to save the markdown preview")
    parser.add_argument("--examples", "-n", type=int, default=5, help="Number of examples of each type to include")
    
    args = parser.parse_args()
    
    # Create the preview
    create_dataset_preview(args.dataset_path, args.output, args.examples)
    
    print(f"Dataset preview created at {args.output}")
    print(f"HTML version available at {args.output.replace('.md', '.html')}")

if __name__ == "__main__":
    main()
