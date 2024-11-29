import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import run_benchmark

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def verify_solution(solution: str, correct_answer: str, problem: str) -> Tuple[str, bool]:
    """Verify a solution and return (answer, is_correct)"""
    answer = extract_answer_from_solution(solution)
    if answer is None:
        return None, False
        
    # Check required keywords
    solution_lower = solution.lower()
    has_required = all(kw in solution_lower for kw in ['problem', 'analysis', 'step'])
    
    if not has_required:
        return answer, False
        
    return answer, await compare_math_answers(answer, correct_answer, problem, verifier_model)

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, best_of: int) -> Optional[Dict]:
    """Process a single example for the standard benchmark"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        prompt = [SystemMessage(content=BENCHMARK_SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        verify_func = lambda sol: verify_solution(sol, correct_answer, example["problem"])
        solutions, best_solution, best_answer, correct_count = await process_attempts(
            solver_model, prompt, best_of, running_id, verify_func)
        
        # Use first attempt if no correct solution found
        solution = best_solution if best_solution is not None else solutions[0]['solution']
        model_answer = best_answer[0] if best_answer is not None else solutions[0]['answer']
        
        # Print results
        success_ratio = f"{correct_count}/{best_of}"
        success_percentage = (correct_count / best_of) * 100
        print(f"\nProblem {running_id + 1}: {success_ratio} ({success_percentage:.1f}%)")
        print(f"Extracted Answer: {correct_answer}")
        print(f"Model's Answer: {model_answer}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'model_answer_raw': model_answer,
            'correct_answer_raw': correct_answer,
            'attempts': {
                'total': len(solutions),
                'correct_count': correct_count
            }
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for benchmarking mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on NuminaMath-CoT dataset')
    await run_benchmark(config, process_example, BENCHMARK_SYSTEM_PROMPT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
