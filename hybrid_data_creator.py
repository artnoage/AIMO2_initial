import os
import re
import time
import random
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *
from bench_utils.verify import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hybrid_creator.log')
    ]
)

# Compile regex patterns once
STEP_NUMBER_PATTERNS = [
    re.compile(r'^.{0,2}(\d+)[.:\)]'),
    re.compile(r'^.{0,2}\((\d+)\)'),
    re.compile(r'^.{0,2}(\d+)\s')
]

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

def validate_step(resp: str, expected_step: Optional[int] = None) -> bool:
    """Validate a solution step"""
    if "[/INST]" in resp:
        return False
    # Check if response has less than 20 words
    word_count = len(resp.split())
    if word_count < 20 or word_count > 100:
        return False
        
    # Check step numbering if expected step is provided
    if expected_step is not None:
        step_mentions = [
            f"step {expected_step}",
            f"step{expected_step}",
            f"({expected_step})",
            f"{expected_step}."
        ]
        if not any(mention.lower() in resp.lower() for mention in step_mentions):
            return False
            
    # Steps should not have multiple step mentions
    step_count = resp.lower().count("step")
    return step_count <= 1

def analyze_solution_quality(solution: str) -> Dict[str, Any]:
    """Analyze various quality metrics of a solution"""
    explanation_patterns = r'because|since|as\s+|explain|due\s+to|results?\s+in|leads?\s+to'
    logical_patterns = r'therefore|thus|hence|consequently|so|accordingly'
    
    return {
        'length': len(solution.split()),
        'has_analysis': bool(re.search(r'analysis|approach|strategy', solution.lower())),
        'step_count': len(re.findall(r'step\s+\d+', solution.lower())),
        'has_boxed': '\\boxed{' in solution,
        'has_equations': bool(re.search(r'\$.*\$', solution)),
        'has_therefore': bool(re.search(logical_patterns, solution.lower())),
        'has_explanation': bool(re.search(explanation_patterns, solution.lower())),
        'formatting_quality': sum([
            '\\boxed{' in solution,
            bool(re.search(r'\$.*\$', solution)),
            bool(re.findall(r'step\s+\d+', solution.lower())),
            bool(re.search(logical_patterns, solution.lower())),
            bool(re.search(explanation_patterns, solution.lower()))
        ])
    }

def calculate_rejected_score(solution: str) -> float:
    """Calculate rejected solution score starting from 0.4 and applying penalties"""
    score = 0.4
    
    # Penalty for no boxed answer
    if '\\boxed{' not in solution:
        score -= 0.2
        
    # Penalty for short solutions
    if len(solution.split()) < 80:
        score -= 0.1
        
    # Penalty for invalid analysis
    if not validate_analysis(solution):
        score -= 0.1
        
    # Penalty for incorrect step numbering
    steps = solution.lower().split("step")
    if len(steps) > 1:  # Only check if there are steps
        found_numbers = []
        missing_numbers = 0
        
        for step in steps[1:]:  # Skip text before first "step"
            number_found = False
            for pattern in STEP_NUMBER_PATTERNS:
                match = pattern.search(step)
                if match:
                    found_numbers.append(int(match.group(1)))
                    number_found = True
                    break
            if not number_found:
                missing_numbers += 1
                logging.debug(f"Missing step number in: {step[:50]}...")
        
        # Check if numbers are sequential starting from 1
        expected_sequence = list(range(1, len(steps)))
        
        # Calculate penalties
        if missing_numbers > 0:
            penalty = min(0.1, 0.02 * missing_numbers)
            score -= penalty
            logging.debug(f"Applied penalty {penalty} for {missing_numbers} missing step numbers")
            
        if found_numbers:
            # Check sequence correctness
            wrong_numbers = sum(1 for a, b in zip(found_numbers, expected_sequence) if a != b)
            if wrong_numbers > 0:
                penalty = min(0.1, 0.02 * wrong_numbers)
                score -= penalty
                logging.debug(f"Applied penalty {penalty} for {wrong_numbers} incorrect step numbers")
            
    return max(0.0, score)  # Ensure non-negative score



async def process_full_solution(example: Dict, solver: any, verifier: any, config: BenchmarkConfig) -> Optional[Tuple[str, str, str, float, float]]:
    """Process example using full solution approach"""
    logs = []
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
        
    rejected_score = calculate_rejected_score(wrong_solution)
    
    # Print detailed logs
    logs.append("\n" + "="*50)
    logs.append("=== Full Solution Approach Details ===")
    logs.append("="*50)
    
    # Success metrics
    success_rate = sum(1 for s in solutions if s['is_correct']) / len(solutions) * 100
    logs.append(f"\n📊 Success Metrics:")
    logs.append(f"✓ Success rate: {success_rate:.1f}%")
    logs.append(f"✓ Attempts for correct solution: {correct_attempt}/{config.best_of}")
    logs.append(f"✓ Attempts for wrong solution: {wrong_attempt}/{config.best_of}")
    logs.append(f"✓ Total attempts: {attempts}/{config.best_of}")
    
    # Scoring details
    logs.append(f"\n💯 Scoring Details:")
    logs.append(f"✓ Chosen solution score: {chosen_score:.3f}")
    logs.append(f"✓ Rejected solution score: {rejected_score:.3f}")
    logs.append(f"✓ Score difference: {(chosen_score - rejected_score):.3f}")
    
    # Verification details
    logs.append(f"\n🔍 Verification Results by Attempt:")
    for idx, sol in enumerate(solutions, 1):
        success_marker = "✅" if sol['is_correct'] else "❌"
        logs.append(f"\nAttempt {idx} {success_marker}")
        logs.append(f"  ├─ Score: {sol['verification_score']}/{sol['verification_steps']}")
        logs.append(f"  ├─ Success Rate: {(sol['verification_score']/sol['verification_steps']*100):.1f}%")
        logs.append(f"  └─ Answer: {sol['answer']}")
    
    return correct_solution, wrong_solution, chosen_score, rejected_score

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using hybrid approach"""
    start_time = time.perf_counter()
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
        
        logs = []
        logs.append("\n" + "="*50)
        logs.append(f"📝 Processing Example {running_id + 1}")
        logs.append("="*50)
        logs.append(f"\n📋 Problem Summary:")
        logs.append(f"├─ ID: {example_id}")
        logs.append(f"├─ Problem: {example['problem'][:200]}...")
        logs.append(f"└─ Expected Answer: {correct_answer}")
        
        if r < 0.3:  # Full solution approach
            logs.append("\nApproach: Full solution")
            result = await process_full_solution(example, solver, verifier, config)
            if result is None:
                return None
            bifurcation_prompt, chosen, rejected, score_chosen, score_rejected = result
            
        else:  # Analysis/Steps approach
            logs.append("\n=== Analysis/Steps Details ===")
            logs.append("Approach: Progressive solution building")
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
            
            logs.append(f"Bifurcation point: After step {n}")
            logs.append(f"Completion attempts planned: {config.completions}")
            
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
            
            logs.append(f"\nCompletion Results:")
            logs.append(f"Chosen path success rate: {score_chosen:.2%}")
            logs.append(f"Rejected path success rate: {score_rejected:.2%}")
            
            if score_chosen == 0 and score_rejected == 0:
                logs.append("❌ Failed: No successful completions for either path")
                print("\n".join(logs))
                return None
                
            if abs(score_chosen - score_rejected)/max(score_chosen, score_rejected) < 0.2:
                logs.append("❌ Failed: Score difference too small (< 20%)")
                print("\n".join(logs))
                return None
                
            logs.append(f"Score difference: {abs(score_chosen - score_rejected)/max(score_chosen, score_rejected):.1%}")
                
            # Swap if rejected has better score
            if score_rejected > score_chosen:
                chosen, rejected = rejected, chosen
                score_chosen, score_rejected = score_rejected, score_chosen

        # Print collected logs
        print("\n".join(logs))
        
        # Return consistent format regardless of approach
        processing_time = time.perf_counter() - start_time
        result = {
            'id': example_id,
            'prompt': {'content': bifurcation_prompt, 'role': 'user'},
            'chosen': {'content': chosen, 'role': 'assistant'},
            'rejected': {'content': rejected, 'role': 'assistant'},
            'score_chosen': score_chosen,
            'score_rejected': score_rejected,
            'quality_metrics': {
                'chosen': analyze_solution_quality(chosen),
                'rejected': analyze_solution_quality(rejected)
            }
        }
        logs.append(f"\n⏱️ Processing Time: {processing_time:.2f}s")
        return result
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        error_category = (
            "timeout" if isinstance(e, TimeoutError)
            else "validation" if isinstance(e, ValueError)
            else "rate_limit" if "rate limit" in str(e).lower()
            else "context_length" if "context length" in str(e).lower()
            else "other"
        )
        error_details = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'error_category': error_category,
            'processing_time': processing_time,
            'example_id': example_id,
            'processing_time': processing_time
        }
        logging.error(f"\n❌ Error processing example {running_id}:")
        logging.error(f"├─ Error type: {error_details['error_type']}")
        logging.error(f"├─ Error message: {error_details['error_message']}")
        logging.error(f"├─ Processing time: {processing_time:.2f}s")
        logging.error(f"└─ Example ID: {example_id}")
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
