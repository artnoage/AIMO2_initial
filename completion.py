import os
import asyncio
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()


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

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int, config: BenchmarkConfig, initial_steps: int = 0) -> Optional[Dict]:
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

                # Create and use appropriate verifier
                verifier = create_verifier(
                    config.verification_type,
                    verifier_model=verifier_model,
                    second_verifier_model=second_verifier_model,
                    tolerance=config.tolerance
                )
                score, total_steps, current_answer = await verifier.verify(
                    complete_solution,
                    correct_answer,
                    example["problem"]
                )
                
                if score == total_steps:
                    correct_count += 1
                
                solutions.append({
                    'solution': complete_solution,
                    'answer': current_answer,
                    'verification_score': score,
                    'verification_steps': total_steps,
                    'is_correct': score == total_steps,
                    'steps_taken': steps_taken,
                    'total_steps': count_solution_steps(complete_solution)
                })
                    
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                solutions.append({
                    'solution': "Error occurred",
                    'answer': None,
                    'is_correct': False,
                    'verification_level': 0,
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
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['verification_level'] == 4 for s in solutions],
            'model_solutions': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'verification_levels': [s['verification_level'] for s in solutions],
            'steps_info': [{
                'steps_taken': s.get('steps_taken', 0),
                'total_steps': s.get('total_steps', 0)
            } for s in solutions]
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for testing hybrid solution generation."""
    config = BenchmarkConfig.from_args('Test hybrid solution generation')
    verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
    
    # Initialize solver model
    solver_model = get_model(ModelOption[config.solver], temp=config.temperature)
    
    
    await run_benchmark(
        config=config,
        process_example_func=process_example,
        solver_model=solver_model,
        verifier_model=verifier_model,
        second_verifier_model=second_verifier_model
    )
    
    if progress_tracker:
        progress_tracker.print_final_stats()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        if progress_tracker:
            progress_tracker.print_final_stats()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        if progress_tracker:
            progress_tracker.print_final_stats()
