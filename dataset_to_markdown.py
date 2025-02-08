import os
import asyncio
import logging
from typing import Dict, List
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import *
from utils.logger import BenchmarkLogger
import time
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> List[Dict]:
    """Process a single example and generate markdown documentation"""
    try:
        logger = BenchmarkLogger()
        
        if not isinstance(example, dict) or 'problem' not in example or 'model_solutions' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None

        # Initialize model and tutor agent
        main = get_model(config, role="main")
        tutor_agent = TutorAgent(main)
        
        # Start markdown content for this example
        markdown_content = []
        markdown_content.append(f"## Example {running_id + 1} (ID: {example_id})\n")
        
        # Add problem
        markdown_content.append("### Problem\n")
        markdown_content.append(f"{example['problem']}\n")
        
        # Process each solution
        for i, solution in enumerate(example['model_solutions'], 1):
            markdown_content.append(f"### Solution {i}\n")
            markdown_content.append("```\n" + solution + "\n```\n")
            
            # Get tutor response
            try:
                tutor_response = await tutor_agent.find_first_wrong_step(example['problem'], solution)
                markdown_content.append("### Tutor's Response\n")
                markdown_content.append("```\n" + tutor_response + "\n```\n")
            except Exception as e:
                markdown_content.append(f"### Error Getting Tutor's Response\n")
                markdown_content.append(f"```\nError: {str(e)}\n```\n")
            
            markdown_content.append("---\n")  # Add separator between solutions
        
        # Add extra separator between examples
        markdown_content.append("\n\n")
        
        # Create results with markdown content
        results = [{
            'id': example_id,
            'data_type': 'markdown',
            'content': '\n'.join(markdown_content)
        }]
        
        logger.append(f"✓ Processed example {running_id + 1}")
        logger.print()
        return results

    except Exception as e:
        logger = BenchmarkLogger()
        logger.append(f"❌ Error processing example {running_id}: {str(e)}")
        logger.print()
        return []

async def main():
    """Main function for generating markdown documentation"""
    config = BenchmarkConfig.from_args('Dataset to Markdown converter')
    
    # Create output directory if it doesn't exist
    os.makedirs("markdown_output", exist_ok=True)
    
    # Initialize markdown file
    timestamp = int(time.time())
    output_file = f"markdown_output/dataset_{timestamp}.md"
    
    # Store all markdown content
    all_markdown_content = []
    all_markdown_content.append("# Dataset Documentation\n\n")
    all_markdown_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Create tracker with content collection callback
    async def collect_results(results: List[Dict]):
        if results:
            for result in results:
                if result['data_type'] == 'markdown':
                    all_markdown_content.append(result['content'])
    
    tracker = ProgressTracker(total_examples=0, config=config)
    tracker.add_result = lambda x: asyncio.create_task(collect_results(x))  # Properly await the coroutine
    await tracker.run_benchmark(process_example_func=process_example)
    
    # Write all content to file at once
    with open(output_file, "w") as f:
        f.write("\n".join(all_markdown_content))
    
    print(f"\n✓ Markdown documentation generated: {output_file}")

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Documentation generation interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Documentation generation failed with error: {e}")
        logger.print()
