import os
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from langchain_core.messages import SystemMessage, HumanMessage
from utils.utils import ModelOption, get_model, extract_answer_from_solution
from utils.augmented_data_handler import handle_augmented_data_file, save_augmented_data, get_existing_ids

# Load environment variables
load_dotenv()
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# System prompts for each role
SOLVER_PROMPT = """You are a mathematical problem solver. When given a problem, solve it step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

In the end provide your final answer inside \\boxed{}"""

JUDGE_PROMPT = """You are a mathematical solution judge. You will be given multiple different solutions to the same problem. Your task is to:
1. Review all solutions carefully
2. Identify the most common answer if there is one
3. Evaluate the reasoning in solutions that arrived at this answer
4. Make a final determination of the most likely correct answer

Provide your final answer inside \\boxed{}"""

VERIFIER_PROMPT = """You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."""

async def get_multiple_solutions(problem: str, solver_model, n_attempts: int) -> List[Dict]:
    """Get multiple solution attempts from the solver model"""
    solutions = []
    prompt = [SystemMessage(content=SOLVER_PROMPT),
              HumanMessage(content=problem)]
    
    for _ in range(n_attempts):
        try:
            response = await solver_model.ainvoke(prompt)
            solution = response.content
            answer = extract_answer_from_solution(solution)
            if answer:
                # Verify this solution immediately
                is_correct = await verify_answer(problem, answer, correct_answer, verifier_model)
                solutions.append({
                    'solution': solution,
                    'answer': answer,
                    'is_correct': is_correct
                })
        except Exception as e:
            print(f"Error getting solution: {e}")
            continue
            
    return solutions

async def get_judge_decision(problem: str, solutions: List[Dict], judge_model) -> Optional[str]:
    """Have the judge model select the best answer from multiple solutions"""
    if not solutions:
        return None
        
    solutions_text = "\n\n".join([
        f"Solution {i+1}:\n{s['solution']}\nExtracted answer: {s['answer']}"
        for i, s in enumerate(solutions)
    ])
    
    prompt = [
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=f"Problem:\n{problem}\n\nMultiple solutions:\n{solutions_text}\n\nWhat is the correct answer?")
    ]
    
    try:
        response = await judge_model.ainvoke(prompt)
        return extract_answer_from_solution(response.content)
    except Exception as e:
        print(f"Error getting judge decision: {e}")
        return None

async def verify_answer(problem: str, model_answer: Optional[str], correct_answer: Optional[str], verifier_model) -> bool:
    """Verify if the model's answer matches the correct answer"""
    if model_answer is None or correct_answer is None:
        return False
        
    prompt = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
    ]
    
    try:
        response = await verifier_model.ainvoke(prompt)
        return response.content.strip().lower() == 'yes'
    except Exception:
        return False

async def process_example(
    example: Dict,
    running_id: int,
    example_id: int,
    solver_model,
    judge_model,
    verifier_model,
    n_attempts: int
) -> Optional[Dict]:
    """Process a single example through all three stages"""
    try:
        # Extract correct answer
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None
            
        # Get multiple solutions
        solutions = await get_multiple_solutions(example['problem'], solver_model, n_attempts)
        if not solutions:
            print(f"Warning: No valid solutions generated for example {running_id}")
            return None
            
        # Get judge's decision
        final_answer = await get_judge_decision(example['problem'], solutions, judge_model)
        if final_answer is None:
            print(f"Warning: Judge could not determine answer for example {running_id}")
            return None
            
        # Verify final answer
        is_correct = await verify_answer(example['problem'], final_answer, correct_answer, verifier_model)
        
        # Print results
        status = '✓' if is_correct else '✗'
        print(f"\nProblem {running_id + 1}: {status}")
        print(f"Number of solutions: {len(solutions)}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Final Answer: {final_answer}")
        print("-" * 80)
        
        # Calculate statistics about initial solutions
        correct_in_initial = any(s['is_correct'] for s in solutions)
        num_correct_initial = sum(1 for s in solutions if s['is_correct'])
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'solutions': solutions,
            'final_answer': final_answer,
            'is_correct': is_correct,
            'correct_in_initial': correct_in_initial,
            'num_correct_initial': num_correct_initial
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description='Multiple sampling benchmark on math problems')
    parser.add_argument('--solver', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Model to use for generating solutions')
    parser.add_argument('--judge', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Model to use for judging solutions')
    parser.add_argument('--verifier', type=str, choices=[model.name for model in ModelOption],
                       default='NEMOTRON', help='Model to use for verifying answers')
    parser.add_argument('--attempts', type=int, default=3,
                       help='Number of solution attempts per problem')
    parser.add_argument('--max-concurrent', type=int, default=4,
                       help='Maximum number of concurrent problems')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (train/validation/test)')
    parser.add_argument('--source', type=str, default='all',
                       help='Filter problems by source')
    parser.add_argument('--dataset', type=str, default='filtered',
                       choices=['original', 'filtered', 'aime'],
                       help='Dataset to use: original (NuminaMath-CoT), filtered (Numina-Olympiads), or aime (AIME validation)')
    args = parser.parse_args()

    # Load dataset
    try:
        if args.dataset == 'original':
            dataset = load_dataset("AI-MO/NuminaMath-CoT", split=args.split)
        elif args.dataset == 'aime':
            dataset = load_dataset("AI-MO/aimo-validation-aime", split=args.split)
        else:
            username = HfApi().whoami()["name"]
            dataset = load_dataset(f"{username}/Numina-Olympiads", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter by source if specified
    if args.source.lower() != 'all':
        dataset = dataset.filter(lambda x: x['source'] == args.source)
    
    # Initialize models
    try:
        solver_model = get_model(ModelOption[args.solver])
        judge_model = get_model(ModelOption[args.judge])
        verifier_model = get_model(ModelOption[args.verifier])
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    # Process examples
    semaphore = asyncio.Semaphore(args.max_concurrent)
    results = []
    
    async def process_with_semaphore(example, running_id):
        async with semaphore:
            return await process_example(
                example, running_id, example['id'],
                solver_model, judge_model, verifier_model,
                args.attempts
            )

    # Create tasks
    tasks = []
    for i, example in enumerate(dataset):
        task = process_with_semaphore({
            'id': example['id'],
            'problem': example['problem'],
            'solution': example['solution']
        }, i)
        tasks.append(task)

    # Process with progress bar
    progress_bar = tqdm(total=len(tasks), desc="Processing examples")
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
        progress_bar.update(1)
    progress_bar.close()

    # Calculate and save results
    if results:
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_count / len(results)) * 100
        
        # Calculate additional statistics
        correct_in_initial_count = sum(1 for r in results if r['correct_in_initial'])
        total_correct_initial = sum(r['num_correct_initial'] for r in results)
        avg_correct_per_problem = total_correct_initial / len(results)
        
        print("\nFinal Results:")
        print(f"Total examples: {len(results)}")
        print(f"Correct final answers: {correct_count}")
        print(f"Final accuracy: {accuracy:.2f}%")
        print(f"Problems with correct answer in initial solutions: {correct_in_initial_count}")
        print(f"Percentage with correct initial: {(correct_in_initial_count/len(results))*100:.2f}%")
        print(f"Average correct solutions per problem: {avg_correct_per_problem:.2f}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join('results', 
            f"multiple_sampling_{args.solver}_{args.judge}_{args.verifier}_{timestamp}.json")
        
        with open(filename, 'w') as f:
            import json
            json.dump({
                'config': vars(args),
                'results': results,
                'summary': {
                    'total': len(results),
                    'correct': correct_count,
                    'accuracy': accuracy
                }
            }, f, indent=2)
        print(f"\nResults saved to {filename}")
    else:
        print("\nNo results generated!")

if __name__ == "__main__":
    asyncio.run(main())


