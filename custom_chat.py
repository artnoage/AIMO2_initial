import os
import asyncio
import logging
import argparse
from typing import Optional, Dict, List
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig, ModelOption
from utils.model_utils import get_model, get_model_response
from utils.logger import BenchmarkLogger
from langchain_core.messages import HumanMessage

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

async def send_custom_message(model_option: str, problem: str, custom_prompt: Optional[str] = None, 
                             temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """
    Send a custom message to the LLM with direct control over the chat template.
    
    Args:
        model_option: The model to use (from ModelOption enum)
        problem: The mathematical problem to solve
        custom_prompt: Optional custom prompt template (if None, uses default SYSTEM_PROMPT)
        temperature: Model temperature setting
        max_tokens: Maximum tokens for model response
        
    Returns:
        The model's response as a string
    """
    logger = BenchmarkLogger()
    
    # Create a minimal config for model initialization
    config = BenchmarkConfig()
    config.main = model_option
    config.main_temp = temperature
    
    # Get the model
    model = get_model(config, role="main")
    
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
    
    logger.append("="*80)
    logger.append(f"📝 Sending custom message to model: {model_option}")
    logger.append("="*80)
    logger.append(f"\n📋 Problem:")
    logger.append(f"{problem[:200]}...")
    logger.append(f"\n📋 Using prompt template:")
    logger.append(f"{prompt_template[:200]}...")
    logger.print()
    
    try:
        # Get response from model
        response = await get_model_response(model, full_prompt, max_tokens=max_tokens)
        
        logger.append("\n📊 Model Response:")
        logger.append("-"*80)
        logger.append(response)
        logger.append("-"*80)
        logger.print()
        
        return response
    except Exception as e:
        logger.append(f"\n❌ Error getting model response: {str(e)}")
        logger.print()
        return f"Error: {str(e)}"

async def main():
    """Main function for sending custom messages to LLMs."""
    parser = argparse.ArgumentParser(description='Send custom messages to LLMs')
    
    # Add arguments
    parser.add_argument('--model', type=str, default='LOCAL_0', 
                        help='Model option (e.g., LOCAL_0, ANTHROPIC_CLAUDE_3_OPUS)')
    parser.add_argument('--problem', type=str, required=True,
                        help='Mathematical problem to solve')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt template (optional)')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='File containing custom prompt template (optional)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Model temperature')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum tokens for model response')
    
    args = parser.parse_args()
    
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
    
    # Send the message
    await send_custom_message(
        model_option=args.model,
        problem=args.problem,
        custom_prompt=custom_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

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
