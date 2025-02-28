import os
import asyncio
import logging
import argparse
import json
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig, ModelOption
from utils.model_utils import get_model, get_model_response
from utils.logger import BenchmarkLogger
from langchain_core.messages import HumanMessage
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Load environment variables
load_dotenv()

# Define system prompt template similar to wait_grpo.py
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

async def send_custom_message(model, problem: str, custom_prompt: Optional[str] = None, 
                             max_tokens: int = 4096) -> str:
    """
    Send a custom message to the LLM with direct control over the chat template.
    
    Args:
        model: The initialized model
        problem: The mathematical problem to solve
        custom_prompt: Optional custom prompt template (if None, uses default SYSTEM_PROMPT)
        max_tokens: Maximum tokens for model response
        
    Returns:
        The model's response as a string
    """
    # Use custom prompt if provided, otherwise use default
    prompt_template = custom_prompt if custom_prompt else SYSTEM_PROMPT
    
    # Construct the full prompt with chat template
    full_prompt = [
        HumanMessage(content=(
            f"<|im_start|>system\n{prompt_template}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        ))
    ]
    
    try:
        # Get response from model
        response = await get_model_response(model, full_prompt, max_tokens=max_tokens)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

async def process_batch(problems: List[Dict[str, Any]], model_option: str, custom_prompt: Optional[str] = None,
                       temperature: float = 0.0, max_tokens: int = 4096, 
                       max_concurrent: int = 3, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Process a batch of problems concurrently.
    
    Args:
        problems: List of problem dictionaries (must contain 'problem' key)
        model_option: The model to use (from ModelOption enum)
        custom_prompt: Optional custom prompt template
        temperature: Model temperature setting
        max_tokens: Maximum tokens for model response
        max_concurrent: Maximum number of concurrent requests
        output_file: Optional file to save results
        
    Returns:
        List of results with model responses
    """
    logger = BenchmarkLogger()
    
    # Create a minimal config for model initialization
    config = BenchmarkConfig()
    config.main = model_option
    config.main_temp = temperature
    
    # Get the model
    model = get_model(config, role="main")
    
    # Create semaphore for limiting concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_problem(problem_dict: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single problem with semaphore control"""
        async with semaphore:
            logger.append(f"Processing problem {index+1}/{len(problems)}: {problem_dict['problem'][:50]}...")
            logger.print()
            
            response = await send_custom_message(
                model=model,
                problem=problem_dict['problem'],
                custom_prompt=custom_prompt,
                max_tokens=max_tokens
            )
            
            # Create result dictionary
            result = problem_dict.copy()
            result['model_response'] = response
            result['timestamp'] = datetime.now().isoformat()
            
            # Save intermediate results if output file specified
            if output_file:
                try:
                    # Append to existing results if file exists
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            all_results = json.load(f)
                        all_results.append(result)
                        with open(output_file, 'w') as f:
                            json.dump(all_results, f, indent=2)
                    else:
                        # Create new file with this result
                        with open(output_file, 'w') as f:
                            json.dump([result], f, indent=2)
                except Exception as e:
                    logger.append(f"Error saving to output file: {str(e)}")
                    logger.print()
            
            return result
    
    # Process all problems concurrently
    tasks = [process_problem(problem, i) for i, problem in enumerate(problems)]
    results = await asyncio.gather(*tasks)
    
    logger.append(f"Completed processing {len(problems)} problems")
    logger.print()
    
    return results

async def main():
    """Main function for batch processing custom messages to LLMs."""
    parser = argparse.ArgumentParser(description='Batch process custom messages to LLMs')
    
    # Add arguments
    parser.add_argument('--model', type=str, default='LOCAL_0', 
                        help='Model option (e.g., LOCAL_0, ANTHROPIC_CLAUDE_3_OPUS)')
    parser.add_argument('--input-file', type=str, required=True,
                        help='JSON file containing problems to process')
    parser.add_argument('--output-file', type=str, default=None,
                        help='JSON file to save results (default: input-file-results.json)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt template (optional)')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='File containing custom prompt template (optional)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Model temperature')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum tokens for model response')
    parser.add_argument('--max-concurrent', type=int, default=3,
                        help='Maximum number of concurrent requests')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}-results.json"
    
    # Load prompt from file if specified
    custom_prompt = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r') as f:
                custom_prompt = f.read()
        except Exception as e:
            print(f"Error reading prompt file: {str(e)}")
            return
    elif args.prompt:
        custom_prompt = args.prompt
    
    # Load problems from input file
    try:
        with open(args.input_file, 'r') as f:
            problems = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return
    
    # Process the batch
    results = await process_batch(
        problems=problems,
        model_option=args.model,
        custom_prompt=custom_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        output_file=args.output_file
    )
    
    # Final save to output file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving final results: {str(e)}")

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Process interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Process failed with error: {e}")
        logger.print()
