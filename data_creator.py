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
    if word_count < 20:
        return False
    # Steps should not have multiple step mentions
    step_count = resp.lower().count("step")
    return step_count <= 1

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    logs = {
        'validation_logs': [],
        'completion_logs': [],
        'path1_logs': [],
        'path2_logs': [],
        'summary_logs': []
    }
    """Process a single example using double analysis approach with multiple completions per analysis"""
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
        verifier_model = None if config.verification_type == 'numeric' else get_model(ModelOption[config.verifier], temp=config.verifier_temp)
        second_verifier_model = None if config.verification_type != 'solution' else get_model(ModelOption[config.second_verifier], temp=config.verifier_temp)
        analysis_agent = AnalysisAgent(solver)
        step_agent = NextStepAgent(solver)
        completion_agent = CompletionAgent(solver)

        # Generate random n with probability proportional to 3^(-n)
        # Calculate normalization constant (sum of 3^(-n) for n=1 to 10)
        norm_const = sum(3**(-i) for i in range(1, 11))
        
        # Generate random number between 0 and 1
        r = random.random()
        cumsum = 0
        n = 1
        
        # Find n where cumulative probability exceeds r
        while n <= 10:
            cumsum += (3**(-n)) / norm_const
            if r <= cumsum:
                break
            n += 1

        
        if n == 1:
            # Get two different analyses with validation
            max_retries = 2
            retry_count = 0
            
            # Get first analysis with validation
            bifurcation_prompt, path_1 = await analysis_agent.generate(example["problem"], return_prompt=True)
            while not validate_analysis(path_1) and retry_count < max_retries:
                print(f"Analysis 1 invalid, retrying... (attempt {retry_count + 1})")
                _, path_1 = await analysis_agent.generate(example["problem"], return_prompt=True)
                retry_count += 1
                
            if not validate_response(path_1):
                print(f"Analysis 1 still invalid after {max_retries} retries, skipping example {running_id}")
                return None
                
            # Get second analysis with validation
            retry_count = 0
            _, path_2 = await analysis_agent.generate(example["problem"], return_prompt=True)
            
            while (path_2 == path_1 or not validate_analysis(path_2)) and retry_count < max_retries:
                print(f"Analysis 2 invalid or matches, retrying... (attempt {retry_count + 1})")
                _, path_2 = await analysis_agent.generate(example["problem"], return_prompt=True)
                retry_count += 1
                
            if path_2 == path_1 or not validate_analysis(path_2):
                print(f"Analysis 2 invalid or matches after {max_retries} retries, skipping example {running_id}")
                return None
                
            response_1 = path_1
            response_2 = path_2
            score_1 = 0
            score_2 = 0
            do_completion_1 = True
            do_completion_2 = True

            logs['validation_logs'].extend([
                "\nPath 1 analysis validation (n=1):",
                f"- Valid analysis: {'Yes' if validate_response(path_1) else 'No'}",
                f"- Initial score: {score_1}",
                f"- Will attempt completion: {'Yes' if do_completion_1 else 'No'}",
                "\nPath 2 analysis validation (n=1):",
                f"- Valid analysis: {'Yes' if validate_response(path_2) else 'No'}",
                f"- Initial score: {score_2}",
                f"- Will attempt completion: {'Yes' if do_completion_2 else 'No'}"
            ])
        else:
            # Common analysis and n-2 steps, then bifurcate
            # Get analysis without prompt
            _, common_analysis = await analysis_agent.generate(example["problem"], return_prompt=True)
            current_solution = common_analysis
            
            # Add n-2 common steps with validation
            for step_num in range(n-2):
                retry_count = 0
                valid_step = False
                while not valid_step and retry_count < 2:
                    _, next_step = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                    # Validate step
                    word_count = len(next_step.split())
                    if word_count >= 20 and "[/INST]" not in next_step and next_step.lower().count("step") <= 1:
                        valid_step = True
                        current_solution += next_step
                    else:
                        retry_count += 1
                        print(f"Step {step_num + 1} invalid, retrying... (attempt {retry_count})")
                
                if not valid_step:
                    print(f"Step {step_num + 1} still invalid after 2 retries, skipping example {running_id}")
                    return None
                    
                # Check if we already have an answer
                if extract_answer_from_solution(current_solution) is not None:
                    print(f"Found answer during common path generation for example {running_id}, skipping")
                    return None
            
            # Generate bifurcation prompt
            # Get two different responses using step agent
            max_retries = 2
            retry_count = 0
            
            # Get first response with validation
            bifurcation_prompt, response_1 = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
            while not validate_step(response_1) and retry_count < max_retries:
                print(f"Response 1 invalid, retrying... (attempt {retry_count + 1})")
                _, response_1 = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                retry_count += 1
                
            if not validate_response(response_1):
                print(f"Response 1 still invalid after {max_retries} retries, skipping example {running_id}")
                return None
                
            # Get second response with validation
            retry_count = 0
            response_2 = await step_agent.generate(example["problem"], current_solution)
            
            while (response_2 == response_1 or not validate_step(response_2)) and retry_count < max_retries:
                print(f"Response 2 invalid or matches, retrying... (attempt {retry_count + 1})")
                response_2 = await step_agent.generate(example["problem"], current_solution)
                retry_count += 1
                
            if response_2 == response_1 or not validate_step(response_2):
                print(f"Response 2 invalid or matches after {max_retries} retries, skipping example {running_id}")
                return None
                
            path_1 = current_solution + response_1
            path_2 = current_solution + response_2
            
            # Check if paths have solutions and validate step responses
            answer_1 = extract_answer_from_solution(path_1)
            answer_2 = extract_answer_from_solution(path_2)
            
            # Initialize scores and completion flags
            score_1 = config.completions if answer_1 is not None else 0
            score_2 = config.completions if answer_2 is not None else 0
            do_completion_1 = answer_1 is None
            do_completion_2 = answer_2 is None

            logs['validation_logs'].extend([
                "\nPath 1 initial check:",
                f"- Found answer: {'Yes' if answer_1 else 'No'}",
                f"- Initial score: {score_1}",
                f"- Need completion: {'Yes' if do_completion_1 else 'No'}",
                "\nPath 2 initial check:",
                f"- Found answer: {'Yes' if answer_2 else 'No'}",
                f"- Initial score: {score_2}",
                f"- Need completion: {'Yes' if do_completion_2 else 'No'}"
            ])
        
        if do_completion_1:
            # Process completions for first analysis
            for _ in range(config.completions):
                try:
                    complete_solution = path_1 + await completion_agent.generate(example["problem"], path_1)
                    verifier = create_verifier(
                        config.verification_type,
                        verifier_model=verifier_model,
                        second_verifier_model=second_verifier_model,
                        tolerance=config.tolerance
                    )
                    score, total_steps, error_msg = await verifier.verify(
                        complete_solution,
                        correct_answer,
                        example["problem"]
                    )
                    if score == total_steps:
                        score_1 += 1
                except Exception as e:
                    logs['completion_logs'].append(f"Error in completion for analysis 1: {str(e)}")
                    error_msg = str(e)

        if do_completion_2:
            # Process completions for second analysis
            for _ in range(config.completions):
                try:
                    complete_solution = path_2 + await completion_agent.generate(example["problem"], path_2)
                    verifier = create_verifier(
                        config.verification_type,
                        verifier_model=verifier_model,
                        second_verifier_model=second_verifier_model,
                        tolerance=config.tolerance
                    )
                    score, total_steps, error_msg = await verifier.verify(
                        complete_solution,
                        correct_answer,
                        example["problem"]
                    )
                    if score == total_steps:
                        score_2 += 1
                except Exception as e:
                    logs['completion_logs'].append(f"Error in completion for analysis 2: {str(e)}")
                    error_msg = str(e)
        score_1=score_1/config.completions
        score_2=score_2/config.completions
        
        
        # Collect summary information
        logs['summary_logs'].extend([
            f"\nExample {running_id + 1} Summary:",
            f"Problem: {example['problem'][:200]}...",
            f"Bifurcation point: Step {n}",
            f"Path 1 final score: {score_1}",
            f"Path 2 final score: {score_2}",
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
            
        # Print summary
        print("\nFINAL SUMMARY:")
        print("\n".join(logs['summary_logs']))
        
        # Skip if scores are very close
        if score_1==0 and score_2==0:
            print(f"Skipping example {running_id+1} - No successful solution")
            return None 
        if  abs(score_1 - score_2)/max(score_1,score_2) < 0.2:
            print(f"Skipping example {running_id+1} - scores too close: {score_1} vs {score_2}")
            return None
            
        # Determine which path had better score
        if score_1 > score_2:
            chosen_response = response_1
            rejected_response = response_2
            score_chosen = score_1
            score_rejected = score_2
        else:
            chosen_response = response_2
            rejected_response = response_1
            score_chosen = score_2
            score_rejected = score_1
            
        return {
            'id': example_id,
            "prompt": {'content': bifurcation_prompt, 'role': 'user'},
            "chosen": {'content': chosen_response, 'role': 'assistant'},
            "rejected": {'content': rejected_response, 'role': 'assistant'},
            'score_chosen': score_chosen,
            'score_rejected': score_rejected
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for benchmarking mathematical problem solving with double analysis."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems using double analysis')
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
