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

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example, sampling until correct answer found or max attempts reached"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {str(running_id)}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {str(running_id)}")
            return None

        model_name = None
        if config.lora_dir:
            model_name = Path(config.lora_dir).name
        elif config.upload_lora:
            latest_lora = get_latest_lora_path()
            if latest_lora:
                model_name = Path(latest_lora).name
                
        solver = get_model(ModelOption[config.solver], temp=config.temperature, model_name=model_name)
        solution_agent = FullSolutionAgent(solver)
        solutions = []
        
        # Create verifier once
        verifier_model = None if config.verification_type == 'numeric' else get_model(
            ModelOption[config.verifier], temp=config.verifier_temp, model_name=model_name)
        second_verifier_model = None if config.verification_type != 'solution' else get_model(
            ModelOption[config.second_verifier], temp=config.verifier_temp, model_name=model_name)
        verifier = create_verifier(
            config.verification_type,
            verifier_model=verifier_model,
            second_verifier_model=second_verifier_model,
            tolerance=config.tolerance
        )
        
        found_correct = False
        found_wrong = False
        correct_attempt = 0
        wrong_attempt = 0
        correct_solution = None
        wrong_solution = None
        
        attempts = 0
        while (not found_correct or not found_wrong) and attempts < config.best_of:
            attempts += 1
            try:
                current_solution = await solution_agent.generate(example["problem"])
                score, total_steps, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                
                is_correct = score == total_steps
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'verification_score': score,
                    'verification_steps': total_steps,
                    'is_correct': is_correct
                })
                
                if is_correct and not found_correct:
                    found_correct = True
                    correct_attempt = attempts
                    correct_solution = current_solution
                elif not is_correct and not found_wrong:
                    found_wrong = True
                    wrong_attempt = attempts
                    wrong_solution = current_solution
                    
            except Exception as e:
                print(f"Error in attempt {attempts} for example {str(running_id)}: {str(e)}")
                solutions.append({
                    'solution': "Error occurred",
                    'answer': None, 
                    'verification_score': 0,
                    'verification_steps': 1,
                    'is_correct': False
                })

        if not found_correct or not found_wrong:
            print(f"Could not find both correct and wrong solutions for example {running_id}")
            return None

        # Calculate scores for ORPO training (range [0,1])
        # Correct solution scoring
        chosen_score = 1.0  # Start at max score
        # Penalty for number of attempts needed (lose up to 0.4)
        attempt_penalty = 0.4 * (correct_attempt-1)/config.best_of
        chosen_score -= attempt_penalty
        
        # Bonus for first try (up to 0.1)
        if correct_attempt == 1:
            chosen_score = min(1.0, chosen_score + 0.1)
            
        # Calculate rejected score with penalties
        rejected_score = calculate_rejected_score(wrong_solution)
            
        # Ensure scores are in [0,1] range
        chosen_score = max(0.0, min(1.0, chosen_score))
        rejected_score = max(0.0, min(1.0, rejected_score))
        
        # Print statistics
        print(f"\nExample {str(running_id + 1)}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Found both solutions: Yes")
        print(f"Correct solution found on attempt: {correct_attempt}")
        print(f"Wrong solution found on attempt: {wrong_attempt}")
        print(f"Total attempts: {attempts}")
        print(f"Chosen score: {chosen_score:.3f}")
        print(f"Rejected score: {rejected_score:.3f}")
        print("-" * 80)
        
        return {
            'id': example_id,
            'prompt': {'content': example['problem'], 'role': 'user'},
            'chosen': {'content': correct_solution, 'role': 'assistant'},
            'rejected': {'content': wrong_solution, 'role': 'assistant'},
            'score_chosen': chosen_score,
            'score_rejected': rejected_score,
            'attempts_chosen': correct_attempt,
            'attempts_rejected': wrong_attempt,
            'total_attempts': attempts,
            'model_solutions': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'verification_scores': [s['verification_score'] for s in solutions],
            'verification_steps': [s['verification_steps'] for s in solutions]
        }
        
    except Exception as e:
        print(f"Error processing example {str(running_id)}: {e}")
        return None

async def main():
    """Main function for sampling until correct solution found."""
    config = BenchmarkConfig.from_args('Sample solutions until correct answer found')
    
    await run_benchmark(
        config=config,
        process_example_func=process_example
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
