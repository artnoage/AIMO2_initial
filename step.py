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

def normalize_latex(text: str) -> str:
    """
    Replace more than two backslashes with two backslashes to fix excessive escaping.
    
    Args:
        text: Input string containing LaTeX notation
        
    Returns:
        Normalized string with consistent backslash escaping
        
    Example:
        >>> normalize_latex(r'\\\\frac{1}{2}')
        '\\\\frac{1}{2}'
    """
    return re.sub(r'\\{3,}', r'\\\\', text)

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, second_verifier_model, best_of: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using sequential agents"""
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
        
        solutions = []
        correct_count = 0
        max_steps = 20  # Maximum steps to prevent infinite loops
        
        for attempt in range(best_of):
            try:
                # Start with analysis
                current_solution = await analysis_agent.generate(example["problem"])
                steps_taken = 0
                has_answer = False
                current_solution = normalize_latex(current_solution.content)
                # Keep adding steps until we get an answer or hit max steps
                while not has_answer and steps_taken < max_steps:
                    steps_taken += 1
                    # Get next step
                    next_step = await step_agent.generate(
                        example["problem"], 
                        current_solution
                    )
                    
                    # Handle AIMessage or string content
                    step_content = next_step.content if hasattr(next_step, 'content') else str(next_step)
                    step_content = normalize_latex(step_content)
                    current_solution = current_solution + step_content
                    
                    # Check if we have an answer
                    has_answer = extract_answer_from_solution(current_solution) is not None
                
                    if has_answer:
                        # Create and use appropriate verifier
                        verifier = create_verifier(
                            config.verification_type,
                            verifier_model=verifier_model,
                            second_verifier_model=second_verifier_model,
                            tolerance=config.tolerance
                        )
                        level, current_answer = await verifier.verify(
                            current_solution,
                            correct_answer,
                            example["problem"]
                        )
                        
                        if level == 4:
                            correct_count += 1
                            
                        solutions.append({
                            'solution': current_solution,
                            'answer': current_answer,
                            'verification_level': level,
                            'is_correct': level == 4,
                            'steps': steps_taken
                        })
                        break
                
                if not has_answer:
                    # If we hit max steps without an answer
                    solutions.append({
                        'solution': current_solution,
                        'answer': None,
                        'is_correct': False,
                        'steps': steps_taken
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
        print(f"Steps taken: {[s['steps'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{best_of}")
        print(f"Success rate: {(correct_count/best_of)*100:.1f}%")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['solution'].split("Problem:")[0].strip() for s in solutions],  # Remove metadata
            'model_answers': [s['answer'] for s in solutions],
            'steps_taken': [s['steps'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'correct_binary': [1 if s['is_correct'] else 0 for s in solutions],
            'model_answer_raw': solutions[0]['answer'],
            'correct_answer_raw': correct_answer,
            'attempts': {
                'total': len(solutions),
                'correct_count': correct_count
            },
            'total_solution_steps': [s['steps'] for s in solutions],
            'solution_type': 'step-by-step'  # Indicate this uses step-by-step generation
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for testing step-by-step solution generation."""
    config = BenchmarkConfig.from_args('Test step-by-step solution generation')
    verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
    verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
    
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
                second_verifier_model=second_verifier_model,
                best_of=best_of,
                config=config
            ),
        config=config
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
