import os
import asyncio
from typing import Optional, Dict, Tuple, Literal
from dotenv import load_dotenv
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import *
from utils.agents import *

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

def count_solution_steps(solution: str) -> int:
    """
    Count the number of steps in a solution by looking for 'Step X' patterns
    Returns the highest step number found to handle missing intermediate steps
    """
    # Look for patterns like "Step 1", "Step 2", etc.
    step_numbers = [
        int(num) for num in re.findall(r'Step\s+(\d+)', solution, re.IGNORECASE)
    ]
    return max(step_numbers) if step_numbers else 0

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int, initial_steps: int = 0) -> Optional[Dict]:
    """Process a single example using hybrid approach: analysis + initial_steps + completion"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Initialize agents
        analysis_agent = AnalysisAgent(solver_model)
        step_agent = NextStepAgent(solver_model)
        completion_agent = CompletionAgent(solver_model)
        
        solutions = []
        correct_count = 0
        
        for attempt in range(best_of):
            try:
                # Start with analysis
                current_solution = await analysis_agent.generate(example["problem"])
                current_solution = current_solution.content
                steps_taken = 0
                complete_solution = current_solution
                for step in range(initial_steps):
                    steps_taken += 1
                    next_step = await step_agent.generate(
                        example["problem"], 
                        HumanMessage(content=current_solution)
                    )
                    step_content = next_step.content if hasattr(next_step, 'content') else str(next_step)
                    current_solution = f"{current_solution}\n\n{step_content}"
                    # Check if we already have an answer
                    if extract_answer_from_solution(current_solution) is not None:
                        print("answer found in step", step + 1, "for attempt", attempt, "in problem", running_id)
                        complete_solution = current_solution
                        break
                else:
                    # Complete the solution if we didn't find an answer in the first two steps
                    steps_taken += 1
                    completion = await completion_agent.generate(example["problem"], current_solution)
                    completion_content = completion.content if hasattr(completion, 'content') else str(completion)
                    complete_solution = completion_content

                # Verify solution using configured verification type
                level, current_answer = await verify_solution(
                    complete_solution,
                    correct_answer,
                    example['problem'],
                    config.verification_type,
                    verifier_model,
                    second_verifier_model,
                    config.tolerance
                )
                
                if level == 4:
                    correct_count += 1
                
                solutions.append({
                    'solution': current_solution,
                    'complete_solution': complete_solution,
                    'answer': current_answer,
                    'verification_level': level,
                    'steps_before_completion': steps_taken,
                    'total_steps': count_solution_steps(complete_solution),
                    'is_correct': level == 4
                })
                    
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                solutions.append({
                    'solution': "Error occurred",
                    'answer': None,
                    'is_correct': False,
                    'steps': 0
                })
        
        # Print statistics
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Steps before completion: {[s['steps_before_completion'] for s in solutions]}")
        print(f"Total solution steps: {[s['total_steps'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{best_of}")
        print(f"Success rate: {(correct_count/best_of)*100:.1f}%")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['complete_solution'].split("Problem:")[0].strip() for s in solutions],  # Remove metadata
            'intermediate_solutions': [s['solution'].split("Problem:")[0].strip() for s in solutions],  # Intermediate steps
            'model_answers': [s['answer'] for s in solutions],
            'steps_before_completion': [s['steps_before_completion'] for s in solutions],
            'total_solution_steps': [s['total_steps'] for s in solutions],
            'verification_levels': [s['verification_level'] for s in solutions],
            'is_correct_list': [s['verification_level'] == 4 for s in solutions],
            'correct_binary': [1 if s['verification_level'] == 4 else 0 for s in solutions],
            'model_answer_raw': solutions[0]['answer'],
            'correct_answer_raw': correct_answer,
            'attempts': {
                'total': len(solutions),
                'correct_count': correct_count
            },
            'steps_taken': [s['steps_before_completion'] for s in solutions],
            'solution_type': 'hybrid'  # Indicate this uses hybrid generation approach
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for testing hybrid solution generation."""
    config = BenchmarkConfig.from_args('Test hybrid solution generation')
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(
        total_examples=config.split_slice.stop if config.split_slice else 0,
        best_of=config.best_of
    )
    
    await run_benchmark(
        config,
        lambda example, running_id, example_id, solver_model, verifier_model, best_of:
            process_example(
                example=example,
                running_id=running_id,
                example_id=example_id,
                solver_model=solver_model,
                verifier_model=verifier_model,
                best_of=best_of
            ),
        progress_tracker=progress_tracker
    )
    
    # Save final results
    progress_tracker.save_results(config.solver, config.split)
    progress_tracker.print_final_stats()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
