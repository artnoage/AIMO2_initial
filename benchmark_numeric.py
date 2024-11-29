import asyncio
from typing import Optional, Dict
from utils.utils import *
from utils.benchmark_utils import run_benchmark
from langchain_core.messages import HumanMessage, SystemMessage
from utils.benchmark_config import *

async def verify_numeric(solution: str, correct_answer: float) -> Tuple[float, bool]:
    """Verify a numeric solution and return (answer, is_correct)"""
    answer = extract_numeric_answer(solution)
    if answer is None:
        return None, False
    return answer, is_answer_correct(answer, correct_answer, tolerance)

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, best_of: int) -> Optional[Dict]:
    """Process a single example for numeric benchmarks"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_numeric_answer(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract numeric answer from solution for example {running_id}")
            return None

        prompt = [SystemMessage(content=NUMERIC_SOLVER_SYSTEM_PROMPT)] + [HumanMessage(content=example["problem"])]
        
        verify_func = lambda sol: verify_numeric(sol, correct_answer)
        solutions, best_solution, best_answer, correct_count = await process_attempts(
            solver_model, prompt, best_of, running_id, verify_func)
        
        solution = best_solution if best_solution is not None else solutions[0]['solution']
        model_answer = best_answer[0] if best_answer is not None else solutions[0]['answer']
        
        # Print statistics
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{best_of}")
        print(f"Success rate: {(correct_count/best_of)*100:.1f}%")
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
    """Main function for benchmarking numeric problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on numeric problems')
    await run_benchmark(config, process_example, NUMERIC_SOLVER_SYSTEM_PROMPT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
