import os
import asyncio
import random
from typing import Optional, Dict, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

def validate_analysis(resp: str) -> bool:
    """Validate an analysis response"""
    if "[/INST]" in resp:
        return False
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 20:
        return False
    # Analysis should mention problem and analysis
    if "problem" not in resp.lower() or "analysis" not in resp.lower():
        return False
    return True

def calculate_rejected_score(solution: str) -> float:
    """Calculate rejected solution score starting from 0.4 and applying penalties"""
    score = 0.4
    
    # Penalty for no boxed answer
    if not any(c in solution for c in ['□', '■', '▢', '▣', '⬚', '▤', '▥', '▦']):
        score -= 0.2
        
    # Penalty for short solutions
    if len(solution.split()) < 80:
        score -= 0.1
        
    # Penalty for invalid analysis
    if not validate_analysis(solution):
        score -= 0.1
        
    return max(0.1, score)  # Ensure minimum score of 0.1

def validate_step(resp: str) -> bool:
    """Validate a solution step"""
    if "[/INST]" in resp:
        return False
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 20 or word_count>100:
        return False
    # Steps should not have multiple step mentions
    step_count = resp.lower().count("step")
    return step_count <= 1

async def process_full_solution(example: Dict, running_id: int, solver: any, verifier: any, config: BenchmarkConfig) -> Optional[Tuple[str, str, float, float]]:
    """Process example using full solution approach"""
    solution_agent = FullSolutionAgent(solver)
    solutions = []
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
                extract_answer_from_solution(example['solution']),
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
            print(f"Error in full solution attempt {attempts}: {str(e)}")
            continue

    if not found_correct or not found_wrong:
        return None

    # Calculate scores
    chosen_score = 1.0 - (0.4 * (correct_attempt-1)/config.best_of)
    if correct_attempt == 1:
        chosen_score = min(1.0, chosen_score + 0.1)
        
    wrong_solution_info = next(s for s in solutions if s['solution'] == wrong_solution)
    rejected_score = calculate_rejected_score(wrong_solution)
    
    # Print detailed logs
    print("\nFULL SOLUTION APPROACH DETAILS:")
    print(f"Correct solution found on attempt: {correct_attempt}")
    print(f"Wrong solution found on attempt: {wrong_attempt}")
    print(f"Total attempts: {attempts}")
    print(f"Chosen score: {chosen_score:.3f}")
    print(f"Rejected score: {rejected_score:.3f}")
    
    return correct_solution, wrong_solution, chosen_score, rejected_score

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using hybrid approach"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Initialize models and verifier
        solver = get_model(ModelOption[config.solver], temp=config.temperature)
        verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
        second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
        verifier = create_verifier(
            config.verification_type,
            verifier_model=verifier_model,
            second_verifier_model=second_verifier_model,
            tolerance=config.tolerance
        )

        # Random approach selection
        r = random.random()
        
        print(f"\nProcessing example {running_id + 1}")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        
        if r < 0.3:  # Full solution approach
            print("\nUsing FULL SOLUTION approach")
            result = await process_full_solution(example, running_id, solver, verifier, config)
            if result is None:
                return None
            chosen, rejected, score_chosen, score_rejected = result
            bifurcation_prompt = example['problem']
            
        else:  # Analysis/Steps approach
            print("\nUsing ANALYSIS/STEPS approach")
            # Calculate step number for bifurcation
            if r < 0.5:  # Analysis only (0.3-0.5 = 0.2 probability)
                n = 1
            else:
                # Exponentially decaying probability for steps 2+
                norm_const = sum(3**(-i) for i in range(1, 11))
                r_scaled = (r - 0.5) / 0.5  # Scale remaining probability space to [0,1]
                cumsum = 0
                n = 1
                while n <= 10:
                    cumsum += (3**(-n)) / norm_const
                    if r_scaled <= cumsum:
                        break
                    n += 1
            
            print(f"Bifurcation at step {n}")
            
            analysis_agent = AnalysisAgent(solver)
            step_agent = NextStepAgent(solver)
            completion_agent = CompletionAgent(solver)
            
            # Process using the analysis/steps approach from data_creator.py
            if n == 1:
                bifurcation_prompt, path_1 = await analysis_agent.generate(example["problem"], return_prompt=True)
                if not validate_analysis(path_1):
                    return None
                    
                _, path_2 = await analysis_agent.generate(example["problem"], return_prompt=True)
                if path_2 == path_1 or not validate_analysis(path_2):
                    return None
                    
                chosen = path_1
                rejected = path_2
                
            else:
                _, common_analysis = await analysis_agent.generate(example["problem"], return_prompt=True)
                current_solution = common_analysis
                
                for step_num in range(n-2):
                    _, next_step = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                    if not validate_step(next_step):
                        return None
                    current_solution += next_step
                    
                    if extract_answer_from_solution(current_solution) is not None:
                        return None
                
                bifurcation_prompt, response_1 = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                if not validate_step(response_1):
                    return None
                    
                response_2 = await step_agent.generate(example["problem"], current_solution)
                if response_2 == response_1 or not validate_step(response_2):
                    return None
                    
                chosen = current_solution + response_1
                rejected = current_solution + response_2
            
            # Calculate scores using completion agent and rejected score penalties
            score_chosen = 0
            score_rejected = calculate_rejected_score(rejected)
            
            for _ in range(config.completions):
                try:
                    complete_solution = chosen + await completion_agent.generate(example["problem"], chosen)
                    score, total_steps, _ = await verifier.verify(complete_solution, correct_answer, example["problem"])
                    if score == total_steps:
                        score_chosen += 1
                except Exception as e:
                    print(f"Error in completion for chosen: {str(e)}")
                    
                try:
                    complete_solution = rejected + await completion_agent.generate(example["problem"], rejected)
                    score, total_steps, _ = await verifier.verify(complete_solution, correct_answer, example["problem"])
                    if score == total_steps:
                        score_rejected += 1
                except Exception as e:
                    print(f"Error in completion for rejected: {str(e)}")
            
            score_chosen = score_chosen / config.completions
            score_rejected = score_rejected / config.completions
            
            if score_chosen == 0 and score_rejected == 0:
                print("No successful solutions")
                return None
                
            if abs(score_chosen - score_rejected)/max(score_chosen, score_rejected) < 0.2:
                print("Scores too close")
                return None
                
            # Swap if rejected has better score
            if score_rejected > score_chosen:
                chosen, rejected = rejected, chosen
                score_chosen, score_rejected = score_rejected, score_chosen

        # Return consistent format regardless of approach
        return {
            'id': example_id,
            'prompt': {'content': bifurcation_prompt, 'role': 'user'},
            'chosen': {'content': chosen, 'role': 'assistant'},
            'rejected': {'content': rejected, 'role': 'assistant'},
            'score_chosen': score_chosen,
            'score_rejected': score_rejected,
            'bifurcation_point': n
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for hybrid approach combining full solution and analysis/steps methods."""
    config = BenchmarkConfig.from_args('Hybrid approach combining full solution and analysis/steps methods')
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
