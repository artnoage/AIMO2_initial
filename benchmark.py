import os
import asyncio
from typing import Optional, Dict, Tuple, Literal
from dotenv import load_dotenv
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import *
from utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

VerificationType = Literal['numeric', 'answer', 'solution']

async def verify_solution(
    solution: str,
    correct_answer: str,
    problem: str,
    verification_type: VerificationType,
    verifier_model=None,
    second_verifier_model=None,
    tolerance: float = 1e-6
) -> Tuple[int, Optional[str]]:
    """
    Verify solution using specified verification method
    Returns:
    - verification_level (0-4)
    - extracted_answer or None
    
    Levels:
    0 - Failed format/extraction check
    1 - Failed verification
    2 - Failed first solution verification (solution type only)
    3 - Failed second verification (solution type only)
    4 - Passed all checks
    """
    model_answer = extract_answer_from_solution(solution)
    if model_answer is None or solution is None:
        return 0, None

    if verification_type == 'numeric':
        try:
            numeric_answer = extract_numeric_answer(solution)
            correct_float = float(correct_answer)
            
            if numeric_answer is None or not isinstance(numeric_answer, (int, float)):
                return 1, model_answer
                
            is_correct = abs(float(numeric_answer) - correct_float) <= tolerance
            return 4 if is_correct else 1, model_answer
        except (ValueError, TypeError):
            return 1, model_answer
            
    elif verification_type == 'answer':
        if not verifier_model:
            raise ValueError("Verifier model required for answer verification")
            
        answer_verifier = AnswerVerifierAgent(verifier_model)
        is_correct = await answer_verifier.verify(problem, solution, correct_answer)
        return 4 if is_correct else 1, model_answer
        
    elif verification_type == 'solution':
        if not verifier_model or not second_verifier_model:
            raise ValueError("Both verifier models required for solution verification")
            
        # First verifier
        solution_verifier = SolutionVerifierAgent(verifier_model)
        if not await solution_verifier.verify(problem, solution):
            return 2, model_answer
            
        # Second verifier
        second_solution_verifier = SolutionVerifierAgent(second_verifier_model)
        if not await second_solution_verifier.verify(problem, solution):
            return 3, model_answer
            
        return 4, model_answer
    
    raise ValueError(f"Unknown verification type: {verification_type}")

def verify_numeric(solution: str, correct_answer: str, tolerance: float = 1e-6) -> Tuple[Optional[float], bool]:
    """Verify solution using numeric comparison"""
    try:
        model_answer = extract_numeric_answer(solution)
        correct_float = float(correct_answer)
        
        if model_answer is None or not isinstance(model_answer, (int, float)):
            return None, False
            
        is_correct = abs(float(model_answer) - correct_float) <= tolerance
        return model_answer, is_correct
    except (ValueError, TypeError):
        return None, False
    
    
async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        solution_agent = FullSolutionAgent(solver_model)
        solutions = []
        correct_count = 0
        best_solution = None
        best_answer = None
        
        for attempt in range(best_of):
            try:
                current_solution = await solution_agent.generate(example["problem"], running_id, attempt)
                
                # Verify solution using configured verification type
                level, current_answer = await verify_solution(
                    current_solution,
                    correct_answer,
                    example["problem"],
                    config.verification_type,
                    verifier_model,
                    second_verifier_model,
                    config.tolerance
                )
                
                if level == 4:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
                        best_answer = current_answer
                        
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'verification_level': level,
                    'is_correct': level == 4
                })
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
            },
            'solution_type': 'complete'  # Indicate this is a complete solution, not step-by-step
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for benchmarking mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems')
    verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
    
    await run_benchmark(
        config,
        lambda example, running_id, example_id, solver_model, verifier_model, best_of:
            process_example(
                example=example,
                running_id=running_id,
                example_id=example_id,
                solver_model=solver_model,
                verifier_model=verifier_model,
                second_verifier_model=second_verifier_model,
                best_of=best_of,
                config=config
            )
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
