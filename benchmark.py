import os
import asyncio
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import run_benchmark
from utils.agents import FullSolutionAgent

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def verify_solution(solution: str, correct_answer: str, problem: str, verifier_model, numeric: bool = False, tolerance: float = 1e-6) -> Tuple[Optional[float], bool]:
    """Verify a solution using either numeric comparison or verifier model"""
    answer = extract_answer_from_solution(solution)
    if answer is None:
        return None, False
        
    if numeric:
        try:
            # Extract and compare numeric answers
            model_answer = extract_numeric_answer(solution)
            correct_float = float(correct_answer)
            
            if model_answer is None or not isinstance(model_answer, (int, float)):
                return None, False
                
            is_correct = abs(float(model_answer) - correct_float) <= tolerance
            return model_answer, is_correct
        except (ValueError, TypeError):
            return None, False
    else:
        # Use verifier model for semantic comparison
        is_correct = await compare_math_answers(answer, correct_answer, problem, verifier_model)
        return answer, is_correct

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, best_of: int, numeric: bool = False, tolerance: float = 1e-6) -> Optional[Dict]:
    """Process a single example with either numeric or semantic verification"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        solution_agent = FullSolutionAgent(solver_model, numeric=numeric)
        verify_func = lambda sol: verify_solution(sol, correct_answer, example["problem"], verifier_model)
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        for attempt in range(best_of):
            try:
                current_solution = await solution_agent.generate(example["problem"], running_id, attempt)
                current_answer, is_correct = await verify_func(current_solution)
                
                if is_correct and current_answer is not None:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        best_answer = current_answer
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                current_solution = "Error occurred"
                current_answer = None
                is_correct = False
            
            solutions.append({
                'solution': current_solution,
                'answer': current_answer,
                'is_correct': is_correct
            })
        
    
        model_answer = best_answer if best_answer is not None else solutions[0]['answer']
        
        # Print statistics
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
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
            'correct_binary': [1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions],
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
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems')
    verifier_model = None if config.numeric else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    
    await run_benchmark(
        config,
        lambda ex, rid, eid, sm, vm, bo: process_example(
            ex, rid, eid, sm, vm, bo,
            numeric=config.numeric,
            tolerance=config.tolerance
        ),
        None,
        verifier_model
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
