import os
import json
import asyncio
import argparse
from typing import Dict, List, Tuple
from utils.utils import ModelOption, get_model
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

SYSTEM_PROMPT = """You are a mathematical solution evaluator. Given a problem and two proposed solutions, you must determine which solution(s) are correct.

RESPOND WITH EXACTLY ONE OF THESE WORDS:
- first (if only the first solution is correct)
- second (if only the second solution is correct)
- both (if both solutions are correct)
- neither (if neither solution is correct)

Base your evaluation on:
1. Mathematical correctness
2. Completeness of the solution
3. Valid reasoning and steps
4. Correct final answer

DO NOT provide any explanation - just output one of the four allowed words."""

async def evaluate_solutions(
    problem: str,
    first_solution: str,
    second_solution: str,
    model
) -> str:
    """Evaluate two solutions and return which one(s) are correct."""
    
    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Problem:\n{problem}\n\nFirst Solution:\n{first_solution}\n\nSecond Solution:\n{second_solution}")
    ]
    
    try:
        response = await model.ainvoke(prompt)
        result = response.content.strip().lower()
        
        # Validate response is one of the allowed values
        if result not in ['first', 'second', 'both', 'neither']:
            print(f"Warning: Invalid model response '{result}', defaulting to 'neither'")
            return 'neither'
            
        return result
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 'neither'

async def process_example(
    example: Dict,
    running_id: int,
    selector_model,
    max_attempts: int = 1
) -> Dict:
    """Process a single example with multiple attempts if needed."""
    
    try:
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            
            result = await evaluate_solutions(
                example['problem'],
                example['accept'],
                example['reject'],
                selector_model
            )
            
            # For DPO data, 'first' should be the correct response since
            # 'accept' is the first solution provided
            is_correct = (result == 'first')
            
            if is_correct or attempts >= max_attempts:
                break
        
        return {
            'id': example['id'],
            'selected': result,
            'is_correct': is_correct,
            'attempts': attempts
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Solution Selection Model Benchmark')
    parser.add_argument('--model', type=str, choices=[model.name for model in ModelOption],
                       default='GEMINI_FLASH', help='Model to use for solution selection')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file containing DPO dataset')
    parser.add_argument('--max-concurrent', type=int, default=4,
                       help='Maximum number of concurrent problems (default: 4)')
    parser.add_argument('--max-attempts', type=int, default=1,
                       help='Maximum attempts per problem (default: 1)')
    
    args = parser.parse_args()
    
    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return
        
    # Load input data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            examples = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
        
    if not examples:
        print("Error: No examples found in input file")
        return
        
    print(f"\nLoaded {len(examples)} examples from {args.input}")
    
    # Initialize model
    selector_model = get_model(ModelOption[args.model], temp=0)
    print(f"\nUsing model: {args.model}")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(example, running_id, selector_model, args.max_attempts)
    
    # Process examples with progress bar
    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(examples)]
    results = []
    
    progress_bar = tqdm(total=len(examples), desc="Processing examples")
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
        progress_bar.update(1)
    progress_bar.close()
    
    if not results:
        print("\nNo examples were successfully processed.")
        return
        
    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = (correct / total) * 100
    
    # Count selection distribution
    selections = {
        'first': sum(1 for r in results if r['selected'] == 'first'),
        'second': sum(1 for r in results if r['selected'] == 'second'),
        'both': sum(1 for r in results if r['selected'] == 'both'),
        'neither': sum(1 for r in results if r['selected'] == 'neither')
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('results', 
                                  f"selection_results_{args.model}_{timestamp}.json")
    
    output_data = {
        'model': args.model,
        'total_examples': total,
        'correct_selections': correct,
        'accuracy': accuracy,
        'selection_distribution': selections,
        'detailed_results': results
    }
    
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print results
    print("\nResults:")
    print(f"Total examples processed: {total}")
    print(f"Correct selections: {correct}/{total} = {accuracy:.2f}%")
    print("\nSelection distribution:")
    for selection, count in selections.items():
        percentage = (count / total) * 100
        print(f"{selection}: {count} ({percentage:.2f}%)")
    
    print(f"\nDetailed results saved to: {results_filename}")
    
    end_time = datetime.now()
    print(f"\nTotal execution time: {end_time - start_time}")
    print(f"Average time per example: {(end_time - start_time).total_seconds() / total:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())