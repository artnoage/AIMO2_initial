import os
import json
import random
random.seed(42)  # Fixed seed for reproducibility
import asyncio
import argparse
from typing import Dict, List
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from utils.utils import ModelOption, get_model
from dotenv import load_dotenv

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def verify_solution(problem: str, solution: str, verifier_model, max_retries: int = 3) -> bool:
    """Verify if a solution is detailed, correct and coherent"""
    prompt = [
        SystemMessage(content="You are a mathematical solution validator. Given a problem and a proposed solution, respond ONLY with 'yes' if the solution is mathematically correct, detailed and coherent, or 'no' if it contains any errors, lacks detail, or has incoherent reasoning. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{problem}\n\nProposed solution:\n{solution}\n\nIs this solution mathematically correct, detailed and coherent?")
    ]
    
    for attempt in range(max_retries):
        try:
            response = await verifier_model.ainvoke(prompt)
            return response.content.strip().lower() == 'yes'
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                print(f"\nError during verification (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                return False

async def save_results(current_results, output_file: str, verifier: str, final=False):
    """Save current results to file"""
    try:
        # Load existing results or start with empty list
        output_data = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)

        # Process current results
        for result in current_results:
            result_id = int(str(result['id']).replace('example_', ''))
            existing_entry = next((item for item in output_data if item["id"] == result_id), None)
            
            if existing_entry:
                existing_entry["verifications"]["verifiers"].append(verifier)
                existing_entry["verifications"]["correctness"].append(result["is_correct"])
                existing_entry["verifications"]["timestamps"].append(datetime.now().isoformat())
            else:
                entry = {
                    "id": result_id,
                    "problem": result["problem"],
                    "model_response": result["model_response"],
                    "solution": result.get("solution", ""),
                    "verifications": {
                        "verifiers": [verifier],
                        "correctness": [result["is_correct"]],
                        "timestamps": [datetime.now().isoformat()]
                    }
                }
                output_data.append(entry)

        # Save all verification results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        if not final:
            print(f"\nIntermediate results saved ({len(current_results)} processed so far)")
    except Exception as e:
        print(f"\nError saving results: {e}")

async def process_examples(examples: List[Dict], verifier_model, sample_size: int, max_concurrent: int, input_file: str, output_file: str, args) -> List[Dict]:
    """Process the randomly selected examples with controlled concurrency"""
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(example, i):
        async with semaphore:
            try:
                is_correct = await verify_solution(
                    example['problem'],
                    example['model_response'],
                    verifier_model
                )
                
                result = {
                    'id': example.get('id', f'example_{i}'),
                    'problem': example['problem'],
                    'model_response': example['model_response'],
                    'solution': example.get('solution', ''),
                    'is_correct': is_correct
                }
                
                # Print minimal progress
                status = '✓' if is_correct else '✗'
                print(f"Example {i}/{sample_size}: {status}", end='\r')
                
                # Save results every 1000 examples
                if i % 1000 == 0:
                    await save_results(results, output_file, args.verifier)
                
                return result
            except Exception as e:
                print(f"\nError processing example {i}: {e}")
                return None
    
    # Create tasks for all examples
    tasks = [process_with_semaphore(ex, i+1) for i, ex in enumerate(examples)]
    
    # Process examples and handle cleaning every 500
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        if result:
            results.append(result)
            
            # Clean incorrect solutions every 500 examples if remove_incorrect is enabled
            if args.remove_incorrect and i % 500 == 0:
                incorrect_ids = {int(str(r['id']).replace('example_', '')) for r in results if not r['is_correct']}
                if incorrect_ids:
                    try:
                        with open(input_file, 'r') as f:
                            data = json.load(f)
                        filtered_data = [ex for ex in data if str(ex['id']) not in incorrect_ids]
                        with open(input_file, 'w') as f:
                            json.dump(filtered_data, f, indent=2)
                        print(f"\nCleaned {len(incorrect_ids)} incorrect solutions at {i} examples")
                    except Exception as e:
                        print(f"\nError during cleaning at {i} examples: {e}")
    
    return results

async def main():
    parser = argparse.ArgumentParser(description='Verify mathematical solutions')
    parser.add_argument('--verifier', type=str, 
                       choices=[model.name for model in ModelOption],
                       required=True,
                       help='Model to use for verification')
    parser.add_argument('--input', type=str, default='combo.json',
                       help='Input JSON file containing problems and solutions')
    parser.add_argument('--output', type=str, default='verification_results.json',
                       help='Output JSON file for verification results')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of examples to verify (default: 100, use -1 for entire dataset)')
    parser.add_argument('--remove_incorrect', action='store_true',
                       help='Remove incorrect solutions from the original dataset')
    parser.add_argument('--max-concurrent', type=int, default=32,
                       help='Maximum number of concurrent verifications (default: 4)')
    args = parser.parse_args()

    # Load and validate input file
    try:
        if not os.path.exists(args.input):
            print(f"Error: Dataset file {args.input} not found. Please specify the correct dataset file.")
            return
            
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Dataset file must contain a list of examples")
            return
            
        if len(data) == 0:
            print("Dataset is empty. Please provide a dataset with examples to verify.")
            return
            
        # If sample_size is -1, use the entire dataset
        if args.sample_size == -1:
            sample_size = len(data)
            selected_examples = data
        else:
            sample_size = min(args.sample_size, len(data))
            selected_examples = data[:sample_size]
    except json.JSONDecodeError:
        print(f"Error: File {args.input} is not valid JSON")
        return
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Initialize verifier model with temperature 0
    try:
        verifier_model = get_model(ModelOption[args.verifier], temp=0)
    except Exception as e:
        print(f"Error initializing verifier models: {e}")
        return

    if args.max_concurrent < 1:
        print("Error: Maximum concurrent verifications must be at least 1")
        return

    print(f"\nVerifying {sample_size} examples with max {args.max_concurrent} concurrent verifications...")
    results = await process_examples(selected_examples, verifier_model, sample_size, args.max_concurrent, args.input, args.output, args)

    # Calculate and display statistics
    if results:
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_count / len(results)) * 100
        print(f"\nResults:")
        print(f"Total examples processed: {len(results)}")
        print(f"Correct solutions: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Save final results
        try:
            await save_results(results, args.output, args.verifier, final=True)
            print(f"\nFinal results saved to {args.output}")

            # Only clean incorrect solutions from input file if requested
            if args.remove_incorrect:
                incorrect_ids = {r['id'] for r in results if not r['is_correct']}
                if incorrect_ids:
                    with open(args.input, 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                    filtered_data = [ex for ex in input_data if str(ex['id']) not in incorrect_ids]
                    with open(args.input, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, indent=2)
                    print(f"\nRemoved {len(incorrect_ids)} incorrect solutions from {args.input}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    asyncio.run(main())
