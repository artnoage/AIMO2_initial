import os
import asyncio
import logging
import argparse
import aiohttp
import json
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
from utils.logger import BenchmarkLogger

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

async def send_custom_message(model_name: str, problem: str, custom_prompt: Optional[str] = None, 
                             temperature: float = 0.0, max_tokens: int = 4096,
                             api_base: str = "http://localhost:8000/v1") -> str:
    """
    Send a custom message to the LLM with direct control over the chat template.
    
    Args:
        model_name: The model name to use
        problem: The mathematical problem to solve
        custom_prompt: Optional custom prompt template (if None, uses default SYSTEM_PROMPT)
        temperature: Model temperature setting
        max_tokens: Maximum tokens for model response
        api_base: Base URL for the API
        
    Returns:
        The model's response as a string
    """
    logger = BenchmarkLogger()
    
    # Use custom prompt if provided, otherwise use default
    prompt_template = custom_prompt if custom_prompt else SYSTEM_PROMPT
    
    # Construct the chat messages with the custom template
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": problem}
    ]
    
    # For direct template manipulation (like in wait_grpo.py)
    # Format the prompt as a single string with the chat template markers
    formatted_prompt = (
        f"<|im_start|>system\n{prompt_template}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    logger.append("="*80)
    logger.append(f"📝 Sending custom message to model: {model_name}")
    logger.append("="*80)
    logger.append(f"\n📋 Problem:")
    logger.append(f"{problem[:200]}...")
    logger.append(f"\n📋 Using prompt template:")
    logger.append(f"{prompt_template[:200]}...")
    logger.print()
    
    # Determine if we're using OpenRouter or a local API
    is_openrouter = "openrouter.ai" in api_base
    
    # Prepare the API request
    if is_openrouter:
        url = "https://openrouter.ai/api/v1/chat/completions"
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    else:
        # Local API (like vLLM or similar)
        url = f"{api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_KEY', 'EMPTY')}"
        }
        
        # For APIs that support direct prompt format
        if "llama" in model_name.lower() or "mistral" in model_name.lower():
            payload = {
                "model": model_name,
                "prompt": formatted_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        else:
            # Standard chat format
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
    
    try:
        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.append(f"\n❌ API Error: {response.status} - {error_text}")
                    logger.print()
                    return f"Error: API returned status {response.status} - {error_text}"
                
                result = await response.json()
                
                # Extract the response content based on API format
                if is_openrouter or "choices" in result:
                    response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    response_text = result.get("output", {}).get("content", "")
                
                logger.append("\n📊 Model Response:")
                logger.append("-"*80)
                logger.append(response_text)
                logger.append("-"*80)
                logger.print()
                
                return response_text
                
    except Exception as e:
        logger.append(f"\n❌ Error getting model response: {str(e)}")
        logger.print()
        return f"Error: {str(e)}"

async def main():
    """Main function for sending custom messages to LLMs."""
    parser = argparse.ArgumentParser(description='Send custom messages to LLMs')
    
    # Add arguments
    parser.add_argument('--model', type=str, default='llama3', 
                        help='Model name (e.g., llama3, mistral, claude-3-opus)')
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
    parser.add_argument('--api-base', type=str, default="http://localhost:8000/v1",
                        help='Base URL for API (default: http://localhost:8000/v1)')
    parser.add_argument('--openrouter', action='store_true',
                        help='Use OpenRouter API instead of local API')
    
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
    
    # Set API base URL if using OpenRouter
    api_base = "https://openrouter.ai/api/v1" if args.openrouter else args.api_base
    
    # Send the message
    await send_custom_message(
        model_name=args.model,
        problem=args.problem,
        custom_prompt=custom_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_base=api_base
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
