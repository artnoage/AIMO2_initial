import os
import json
import asyncio
import argparse
from typing import Optional, Dict
from datetime import datetime
from utils.utils import ModelOption, get_model, extract_answer_from_solution
from langchain_core.messages import HumanMessage, SystemMessage
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv

SYSTEM_PROMPT="""You are a precise mathematical problem solver. You will be given a problem to solve.

DO:
▪ List applicable theorems/techniques upfront
▪ If possible each step must contain a justification. 
▪ Use LaTeX notation

FORMAT:

**Problem Analysis and Approach**:
1. Start by categorizing the problem (e.g., "This is an inequality problem involving algebraic identities" or "This is a combinatorial proof").
2. List specific tools or theorems that will guide your solution (e.g., "AM-GM inequality," "Basic algebraic manipulations").

**PROOF**:
Example format for each step:
Given: \\( a, b, c > 0 \\) and \\( a + b + c = 3 \\). Prove that \\( abc \\leq 1 \\).

Step 1. By the AM-GM inequality, \\( \\frac{a + b + c}{3} \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Apply AM-GM inequality to \\( a, b, c \\)]  
Step 2. Substituting \\( a + b + c = 3 \\), we get \\( 1 \\geq \\sqrt[3]{abc} \\) \\hspace{10pt} [Replace with given sum condition]  
Step 3. Cube both sides to eliminate the root: \\( 1 \\geq abc \\) \\hspace{10pt} [Cube both sides to solve for \\( abc \\)]  
Step 4. Thus, \\( abc \\leq 1 \\), as required.  

For each step, clearly state the action, use concise LaTeX notation, and provide a justification in brackets.

**ANSWER**:
\\(\\boxed{\\text{result}}\\) """
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Load environment variables from .env file
load_dotenv()


async def verify_answer(problem: str, model_answer: str, correct_answer: str, verifier_model, verifier_name: str) -> bool:
    """Use a verifier model to compare two mathematical answers"""
    if model_answer is None or correct_answer is None:
        return False
        
    comparison_prompt = [
        SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
        HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
    ]
    
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await asyncio.wait_for(
                verifier_model.ainvoke(comparison_prompt),
                timeout=300  # 5 minutes timeout
            )
            result = response.content.strip().lower() == 'yes'
            print(f"{verifier_name} verdict: {'correct' if result else 'incorrect'}")
            return result
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"{verifier_name} verification failed after {max_retries} attempts")
                return False
            print(f"Connection error during {verifier_name} verification. Retrying... ({retry_count}/{max_retries})")
            await asyncio.sleep(1)
    return False

async def process_example(
    example: Dict,
    running_id: int,
    example_id: int,
    solver_model,
    verifier_models: Dict[str, any]
) -> Optional[Dict]:
    """Process a single example using one solver and multiple verifiers"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        # Extract the correct answer
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Get solver's answer
        prompt = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=example["problem"])
        ]
        
        # Try up to 3 times in case of connection errors
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    solver_model.ainvoke(prompt),
                    timeout=300
                )
                model_answer = extract_answer_from_solution(response.content)
                break
            except Exception as e:
                if retry == max_retries - 1:
                    print(f"Failed to get solver response after {max_retries} attempts")
                    return None
                print(f"Error getting solver response, attempt {retry + 1}/{max_retries}")
                await asyncio.sleep(1)
                continue
                
        if model_answer is None:
            print(f"Warning: Could not extract answer from solver response for example {running_id}")
            return None

        # Get verification results from each verifier
        verifier_results = {}
        for verifier_name, verifier_model in verifier_models.items():
            result = await verify_answer(
                example["problem"],
                model_answer,
                correct_answer,
                verifier_model,
                verifier_name
            )
            verifier_results[verifier_name] = result
            
        # Print results for this example
        print(f"\nProblem {running_id + 1}:")
        print(f"Solver's answer: {model_answer}")
        print(f"Correct answer: {correct_answer}")
        for verifier, result in verifier_results.items():
            print(f"{verifier}: {'Correct' if result else 'Incorrect'}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'verifier_results': verifier_results
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description='Compare multiple verifiers on math problems')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL', help='Model to use for solving problems')
    parser.add_argument('--verifier1', type=str, choices=[model.name for model in ModelOption],
                       default='LOCAL', help='First verifier model')
    parser.add_argument('--verifier2', type=str, choices=[model.name for model in ModelOption],
                       default='GEMINI_FLASH', help='Second verifier model')
    parser.add_argument('--verifier3', type=str, choices=[model.name for model in ModelOption],
                       default='CODER', help='Third verifier model')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source (default: all)')
    parser.add_argument('--max-concurrent', type=int, default=128,
                       help='Maximum number of concurrent problems')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Temperature for solver model generation')
    args = parser.parse_args()

    if args.max_concurrent < 1:
        print("Error: Maximum concurrent problems must be at least 1")
        return

    # Load dataset
    try:
        username = HfApi().whoami()["name"]
        dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    dataset = dataset.shuffle(seed=42)
    print(f"\nDataset size: {len(dataset)} examples")

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    # Initialize models
    try:
        solver_model = get_model(ModelOption[args.solver], temp=args.temperature)
        verifier_models = {
            'verifier1': get_model(ModelOption[args.verifier1], temp=0),
            'verifier2': get_model(ModelOption[args.verifier2], temp=0),
            'verifier3': get_model(ModelOption[args.verifier3], temp=0)
        }
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    print(f"\nComparing verifiers on {args.split} split...")
    print(f"Solver: {args.solver}")
    print(f"Verifier 1: {args.verifier1}")
    print(f"Verifier 2: {args.verifier2}")
    print(f"Verifier 3: {args.verifier3}")

    # Process examples with controlled concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(
                example, running_id, example['id'],
                solver_model, verifier_models
            )

    tasks = [process_with_semaphore(ex, i) for i, ex in enumerate(dataset)]
    
    # Process all examples with progress bar
    results = []
    progress_bar = tqdm(total=len(dataset), desc="Processing examples")
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
        progress_bar.update(1)
        
        # Print intermediate statistics every 100 examples
        if len(results) % 100 == 0 and results:
            print("\nIntermediate Statistics:")
            verifier_agreements = {name: sum(1 for r in results if r['verifier_results'][name])
                                 for name in verifier_models.keys()}
            for name, count in verifier_agreements.items():
                print(f"{name} agreed with correct answer: {count}/{len(results)} = {(count/len(results))*100:.2f}%")
            
            # Calculate agreement between verifiers
            print("\nVerifier Agreement Matrix:")
            for v1 in verifier_models.keys():
                for v2 in verifier_models.keys():
                    if v1 < v2:  # Only print each pair once
                        agreements = sum(1 for r in results 
                                      if r['verifier_results'][v1] == r['verifier_results'][v2])
                        agreement_rate = (agreements / len(results)) * 100
                        print(f"{v1} vs {v2}: {agreement_rate:.2f}% agreement")
            print("-" * 80)
    
    progress_bar.close()

    if not results:
        print("\nNo examples were successfully processed.")
        return

    # Calculate final statistics
    print("\nFinal Statistics:")
    
    # Individual verifier accuracy
    print("\nVerifier Accuracy:")
    verifier_agreements = {name: sum(1 for r in results if r['verifier_results'][name])
                         for name in verifier_models.keys()}
    for name, count in verifier_agreements.items():
        print(f"{name} agreed with correct answer: {count}/{len(results)} = {(count/len(results))*100:.2f}%")
    
    # Agreement between verifiers
    print("\nVerifier Agreement Matrix:")
    agreement_matrix = {}
    for v1 in verifier_models.keys():
        for v2 in verifier_models.keys():
            if v1 < v2:  # Only calculate each pair once
                agreements = sum(1 for r in results 
                               if r['verifier_results'][v1] == r['verifier_results'][v2])
                agreement_rate = (agreements / len(results)) * 100
                agreement_matrix[f"{v1}_vs_{v2}"] = agreement_rate
                print(f"{v1} vs {v2}: {agreement_rate:.2f}% agreement")
    
    # Unanimous agreement cases
    unanimous_correct = sum(1 for r in results 
                          if all(r['verifier_results'].values()))
    unanimous_incorrect = sum(1 for r in results 
                            if not any(r['verifier_results'].values()))
    print(f"\nUnanimous agreement (all correct): {unanimous_correct}/{len(results)} = {(unanimous_correct/len(results))*100:.2f}%")
    print(f"Unanimous agreement (all incorrect): {unanimous_incorrect}/{len(results)} = {(unanimous_incorrect/len(results))*100:.2f}%")

    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join('results', 
                                  f"verifier_comparison_{timestamp}.json")
    
    output_data = {
        'run_parameters': {
            'solver': args.solver,
            'verifier1': args.verifier1,
            'verifier2': args.verifier2,
            'verifier3': args.verifier3,
            'split': args.split,
            'source': args.source,
            'max_concurrent': args.max_concurrent,
            'temperature': args.temperature
        },
        'statistics': {
            'total_examples': len(results),
            'verifier_accuracy': verifier_agreements,
            'agreement_matrix': agreement_matrix,
            'unanimous': {
                'all_correct': unanimous_correct,
                'all_incorrect': unanimous_incorrect
            }
        },
        'timing': {
            'total_duration_seconds': (datetime.now() - start_time).total_seconds()
        },
        'detailed_results': results
    }
    
    with open(results_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {results_filename}")
    print(f"Total execution time: {datetime.now() - start_time}")

if __name__ == "__main__":
    asyncio.run(main())
