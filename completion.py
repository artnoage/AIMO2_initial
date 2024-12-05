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

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using hybrid approach: analysis + initial_steps + completion"""
    logs = {
        'validation_logs': [],
        'completion_logs': [],
        'step_logs': [],
        'summary_logs': []
    }
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Initialize agents
        solver = get_model(ModelOption[config.solver], temp=config.temperature)
        analysis_agent = AnalysisAgent(solver)
        step_agent = NextStepAgent(solver)
        completion_agent = CompletionAgent(solver)
        
        solutions = []
        correct_count = 0
        
        for attempt in range(config.best_of):
            try:
                # Start with analysis
                current_solution = await analysis_agent.generate(example["problem"])
                steps_taken = 0
                for step in range(config.initial_steps):
                    steps_taken += 1
                    next_step = await step_agent.generate(example["problem"], current_solution)
                    current_solution = current_solution + next_step
                    # Check if we already have an answer
                    if extract_answer_from_solution(current_solution) is not None:
                        print("answer found in step", step + 1, "for attempt", attempt, "in problem", running_id)
                        complete_solution = current_solution
                        break
                else:
                    # Complete the solution if we didn't find an answer in the first two steps
                    steps_taken += 1
                    complete_solution  = current_solution+ await completion_agent.generate(example["problem"], current_solution)

                # Create and use appropriate verifier
                verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
                second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
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
                    'verification_score': 0,
                    'verification_steps': 1,
                    'steps_taken': 0,
                    'total_steps': 0
                })
        
        # Collect summary information
        logs['summary_logs'].extend([
            f"\nExample {running_id + 1}:",
            f"Problem: {example['problem'][:200]}...",
            f"Correct answer: {correct_answer}",
            f"Model answers: {[s['answer'] for s in solutions]}",
            f"Steps taken: {[s.get('steps_taken', 0) for s in solutions]}",
            f"Total steps: {[s.get('total_steps', 0) for s in solutions]}",
            f"Correct/incorrect: {[1 if s['is_correct'] else 0 for s in solutions]}",
            f"Correct solutions: {correct_count}/{config.best_of}",
            f"Success rate: {(correct_count/config.best_of)*100:.1f}%",
            "-" * 80
        ])

        # Print all logs in organized sections
        print("\n" + "="*50)
        print(f"COMPLETE LOG FOR EXAMPLE {running_id + 1}")
        print("="*50)
        
        # Print validation logs
        if logs['validation_logs']:
            print("\nVALIDATION DETAILS:")
            print("\n".join(logs['validation_logs']))
            
        # Print completion logs
        if logs['completion_logs']:
            print("\nCOMPLETION PROCESS DETAILS:")
            print("\n".join(logs['completion_logs']))
            
        # Print step logs
        if logs['step_logs']:
            print("\nSTEP PROCESS DETAILS:")
            print("\n".join(logs['step_logs']))
            
        # Print summary
        print("\nFINAL SUMMARY:")
        print("\n".join(logs['summary_logs']))
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'model_solutions': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'verification_scores': [s['verification_score'] for s in solutions],
            'verification_steps': [s['verification_steps'] for s in solutions],
            'steps_info': [{
                'steps_taken': s.get('steps_taken', 0),
                'total_steps': s.get('total_steps', 0)
            } for s in solutions]
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for benchmarking mathematical problem solving."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems')
    
    await run_benchmark(
        config=config,
        process_example_func=process_example
    )

if __name__ == "__main__":
    progress_tracker = None  # Will be initialized in run_benchmark
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if progress_tracker:
            progress_tracker.print_final_stats()
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        if progress_tracker:
            progress_tracker.print_final_stats()
