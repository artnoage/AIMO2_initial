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
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hybrid_creator.log', mode='w')
    ]
)

# Ensure all handlers use the same formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

# Set logging level for specific loggers
logging.getLogger('hybrid_creator').setLevel(logging.DEBUG)

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

def validate_solution(solution: str) -> Tuple[bool, str]:
    """
    Validate a complete solution against all required criteria.
    Returns (is_valid, reason) tuple.
    """
    # Check for analysis section
    if "analysis" not in solution.lower():
        return False, "Missing analysis section"
    
    # Check analysis length
    analysis_parts = [p for p in solution.lower().split("step") if "analysis" in p.lower()]
    if analysis_parts and len(analysis_parts[0].split()) < 20:
        return False, "Analysis section too short (< 20 words)"
        
    # Check for boxed answer
    if "\\boxed{" not in solution:
        return False, "Missing boxed answer"
        
    # Split into steps and validate each
    steps = solution.lower().split("step")[1:]  # Skip text before first "step"
    if not steps:
        return False, "No numbered steps found"
        
    # Track step numbers found
    found_numbers = []
    
    for i, step in enumerate(steps, 1):
        # Check step length
        step_words = len(step.split())
        if step_words < 18:
            return False, f"Step {i} too short ({step_words} words)"
        if step_words > 100:
            return False, f"Step {i} too long ({step_words} words)"
            
        # Check step numbering
        number_found = False
        for pattern in STEP_NUMBER_PATTERNS:
            match = pattern.search(step)
            if match:
                found_numbers.append(int(match.group(1)))
                number_found = True
                break
        if not number_found:
            return False, f"Missing number for step {i}"
            
    # Verify sequential step numbers
    expected_numbers = list(range(1, len(steps) + 1))
    if found_numbers != expected_numbers:
        return False, f"Steps not properly numbered. Found {found_numbers}, expected {expected_numbers}"
        
    return True, "Solution valid"

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



async def process_full_solution(example: Dict, solver: any, verifier: any, config: BenchmarkConfig) -> Optional[Tuple[str, str, str, float, float, str]]:
    """Process example using full solution approach"""
    logs = []
    solution_agent = FullSolutionAgent(solver) 
    bifurcation_prompt = None
    found_correct = False
    found_wrong = False
    correct_attempt = 0
    wrong_attempt = 0
    correct_solution = None
    wrong_solution = None
    total_solution_attempts = 0
    
    attempts = 0
    while (not found_correct or not found_wrong) and attempts < config.best_of:
        attempts += 1
        try:
            retry_count = 0
            while retry_count < 3:  # Try up to 3 times for each attempt
                total_solution_attempts += 1
                if bifurcation_prompt is None:
                    bifurcation_prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
                else:
                    current_solution = await solution_agent.generate(example["problem"])
                
                # First validate solution structure
                is_valid, validation_reason = validate_solution(current_solution)
                if not is_valid:
                    retry_count += 1
                    logs.append(f"Attempt {attempts}.{retry_count} validation failed: {validation_reason}")
                    if retry_count < 3:
                        continue
                    else:
                        logs.append(f"Failed all 3 retries for attempt {attempts}")
                        break
                else:
                    logs.append(f"✓ Attempt {attempts}.{retry_count + 1} passed validation")
                    break  # Valid solution found, exit retry loop
                
            # Then verify correctness
            score, total_steps, current_answer = await verifier.verify(
                current_solution,
                extract_answer_from_solution(example['solution']),
                example["problem"]
            )
            
            is_correct = score == total_steps and is_valid
            if is_correct and not found_correct:
                found_correct = True
                correct_attempt = attempts
                correct_solution = current_solution
                logs.append(f"✓ Found correct solution on attempt {attempts} (try {retry_count + 1})")
                logs.append(f"  Total solution attempts: {total_solution_attempts}")
            elif not is_correct and not found_wrong:
                found_wrong = True
                wrong_attempt = attempts
                wrong_solution = current_solution
                logs.append(f"✗ Found incorrect solution on attempt {attempts} (try {retry_count + 1})")
                
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
    logs.append(f"\n📊 Success Metrics:")
    logs.append(f"✓ Found correct solution on attempt: {correct_attempt}/{config.best_of}")
    logs.append(f"✓ Found wrong solution on attempt: {wrong_attempt}/{config.best_of}")
    logs.append(f"✓ Total attempts needed: {attempts}/{config.best_of}")
    logs.append(f"✓ Success rate: {(found_correct/attempts)*100:.1f}%")
    logs.append(f"✓ Failure rate: {(found_wrong/attempts)*100:.1f}%")
    logs.append(f"✓ Average attempts until correct: {correct_attempt:.1f}")
    
    # Solution quality metrics
    logs.append(f"\n📝 Solution Quality:")
    correct_quality = analyze_solution_quality(correct_solution)
    wrong_quality = analyze_solution_quality(wrong_solution)
    
    logs.append(f"✓ Correct solution:")
    logs.append(f"  ├─ Length: {correct_quality['length']} words")
    logs.append(f"  ├─ Steps: {correct_quality['step_count']}")
    logs.append(f"  ├─ Has equations: {'Yes' if correct_quality['has_equations'] else 'No'}")
    logs.append(f"  └─ Format score: {correct_quality['formatting_quality']}/5")
    
    logs.append(f"✓ Wrong solution:")
    logs.append(f"  ├─ Length: {wrong_quality['length']} words")
    logs.append(f"  ├─ Steps: {wrong_quality['step_count']}")
    logs.append(f"  ├─ Has equations: {'Yes' if wrong_quality['has_equations'] else 'No'}")
    logs.append(f"  └─ Format score: {wrong_quality['formatting_quality']}/5")
    
    # Scoring details
    logs.append(f"\n💯 Scoring Details:")
    logs.append(f"✓ Chosen solution score: {chosen_score:.3f}")
    logs.append(f"✓ Rejected solution score: {rejected_score:.3f}")
    logs.append(f"✓ Score difference: {(chosen_score - rejected_score):.3f}")
    logs.append(f"✓ Relative improvement: {((chosen_score - rejected_score)/rejected_score)*100:.1f}%")
    
    return bifurcation_prompt, correct_solution, wrong_solution, chosen_score, rejected_score

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
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        
        # Problem details
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Approach info
        logs.append(f"\n🔄 Processing Details:")
        logs.append(f"├─ Strategy: {'Full solution' if r < 0.3 else 'Progressive building'}")
        
        if r < 0.3:  # Full solution approach
            result = await process_full_solution(example, solver, verifier, config)
            if result is None:
                return None
            bifurcation_prompt, path1, path2, score_path1, score_path2, solution_logs = result
            
        else:  # Analysis/Steps approach
            logs.append("\n=== Analysis/Steps Details ===")
            logs.append("Approach: Progressive solution building")
            
            # Determine bifurcation point
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
            
            logs.append(f"└─ Bifurcation: After step {n}")
            logs.append(f"Completion attempts planned: {config.completions}")
            
            analysis_agent = AnalysisAgent(solver)
            step_agent = NextStepAgent(solver)
            completion_agent = CompletionAgent(solver)
            
            # Process using the analysis/steps approach from data_creator.py
            if n == 1:
                # Try up to 3 times for path_1
                for retry in range(3):
                    bifurcation_prompt, path_1 = await analysis_agent.generate(example["problem"], return_prompt=True)
                    if validate_analysis(path_1):
                        break
                    logs.append(f"Analysis validation failed for path_1 (retry {retry + 1}/3)")
                    if retry == 2:  # All retries failed
                        logs.append("Failed all retries for path_1 analysis")
                        return None
                
                # Try up to 3 times for path_2
                for retry in range(3):
                    _, path_2 = await analysis_agent.generate(example["problem"], return_prompt=True)
                    if path_2 != path_1 and validate_analysis(path_2):
                        break
                    logs.append(f"Analysis validation failed for path_2 (retry {retry + 1}/3)")
                    if retry == 2:  # All retries failed
                        logs.append("Failed all retries for path_2 analysis")
                        return None
                    
                path1 = path_1
                path2 = path_2
                
            else:
                # Generate initial analysis
                _, common_analysis = await analysis_agent.generate(example["problem"], return_prompt=True)
                if extract_answer_from_solution(common_analysis) is not None:
                    logs.append("❌ Analysis contained premature answer - dropping example")
                    return None
                    
                current_solution = common_analysis
                
                # Generate intermediate steps
                for step_num in range(n-2):
                    step_added = False
                    for retry in range(3):
                        _, next_step = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                        if validate_step(next_step):
                            test_solution = current_solution + next_step
                            premature_answer = extract_answer_from_solution(test_solution)
                            if premature_answer is None:
                                current_solution = test_solution
                                step_added = True
                                break
                            else:
                                logs.append(f"Step {step_num + 1} generated premature answer (retry {retry + 1}/3)")
                        else:
                            logs.append(f"Step {step_num + 1} validation failed (retry {retry + 1}/3)")
                    
                    if not step_added:
                        logs.append(f"Failed all retries for step {step_num + 1}")
                        return None
                
                # Generate bifurcation paths
                bifurcation_prompt, response_1 = await step_agent.generate(example["problem"], current_solution, return_prompt=True)
                if not validate_step(response_1):
                    return None
                
                path1 = current_solution + response_1
                answer1 = extract_answer_from_solution(path1)
                
                if answer1 is not None:
                    # First path found answer - verify it
                    score, total_steps, _ = await verifier.verify(path1, correct_answer, example["problem"])
                    if score == total_steps:
                        logs.append("✓ First path found correct answer at bifurcation - using maximum score")
                        return {
                            'id': example_id,
                            'prompt': {'content': bifurcation_prompt, 'role': 'user'},
                            'chosen': {'content': path1, 'role': 'assistant'},
                            'rejected': {'content': current_solution, 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0,
                            'bifurcation_prompt': bifurcation_prompt,
                            'quality_metrics': {
                                'chosen': analyze_solution_quality(path1),
                                'rejected': analyze_solution_quality(current_solution)
                            }
                        }
                
                response_2 = await step_agent.generate(example["problem"], current_solution)
                if response_2 == response_1 or not validate_step(response_2):
                    return None
                
                path2 = current_solution + response_2
                answer2 = extract_answer_from_solution(path2)
                
                if answer2 is not None:
                    # Second path found answer - verify it
                    score, total_steps, _ = await verifier.verify(path2, correct_answer, example["problem"])
                    if score == total_steps:
                        logs.append("✓ Second path found correct answer at bifurcation - using maximum score")
                        return {
                            'id': example_id,
                            'prompt': {'content': bifurcation_prompt, 'role': 'user'},
                            'chosen': {'content': path2, 'role': 'assistant'},
                            'rejected': {'content': current_solution, 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0,
                            'bifurcation_prompt': bifurcation_prompt,
                            'quality_metrics': {
                                'chosen': analyze_solution_quality(path2),
                                'rejected': analyze_solution_quality(current_solution)
                            }
                        }
            
            # Calculate scores using completion agent and rejected score penalties
            successful_path1 = 0
            successful_path2 = 0
            
            logs.append("\n🔍 Completion Attempts:")
            for attempt in range(config.completions):
                logs.append(f"\nAttempt {attempt + 1}/{config.completions}:")
                
                # Path 1 completion
                try:
                    complete_solution = path1 + await completion_agent.generate(example["problem"], path1)
                    is_valid, validation_reason = validate_solution(complete_solution)
                    logs.append(f"Path 1:")
                    logs.append(f"├─ Validation: {'✓' if is_valid else '✗'}")
                    logs.append(f"├─ Reason: {validation_reason}")
                    
                    if is_valid:
                        score, total_steps, _ = await verifier.verify(complete_solution, correct_answer, example["problem"])
                        logs.append(f"├─ Verification Score: {score}/{total_steps}")
                        if score == total_steps:
                            successful_path1 += 1
                            logs.append(f"└─ Success! ({successful_path1} total successes)")
                        else:
                            logs.append(f"└─ Failed verification")
                    else:
                        logs.append(f"└─ Failed validation")
                except Exception as e:
                    logs.append(f"└─ Error: {str(e)}")
                    
                # Path 2 completion
                try:
                    complete_solution = path2 + await completion_agent.generate(example["problem"], path2)
                    is_valid, validation_reason = validate_solution(complete_solution)
                    logs.append(f"Path 2:")
                    logs.append(f"├─ Validation: {'✓' if is_valid else '✗'}")
                    logs.append(f"├─ Reason: {validation_reason}")
                    
                    if is_valid:
                        score, total_steps, _ = await verifier.verify(complete_solution, correct_answer, example["problem"])
                        logs.append(f"├─ Verification Score: {score}/{total_steps}")
                        if score == total_steps:
                            successful_path2 += 1
                            logs.append(f"└─ Success! ({successful_path2} total successes)")
                        else:
                            logs.append(f"└─ Failed verification")
                    else:
                        logs.append(f"└─ Failed validation")
                except Exception as e:
                    logs.append(f"└─ Error: {str(e)}")
            
            # Calculate success rates as ratios
            score_path1 = successful_path1 / config.completions
            score_path2 = successful_path2 / config.completions
            
            # Calculate relative scores if either has non-zero success
            if score_path1 == 0 and score_path2 == 0:
                logs.append("❌ Failed: No successful completions for either path")
                print("\n".join(logs))
                return {
                    'id': example_id,
                    'status': 'rejected',
                    'reason': 'No successful completions for either path',
                    'processing_time': time.perf_counter() - start_time,
                    'logs': "\n".join(logs)
                }

            max_score = max(score_path1, score_path2)
            relative_path1 = score_path1 / max_score if max_score > 0 else 0
            relative_path2 = score_path2 / max_score if max_score > 0 else 0
            relative_diff = abs(relative_path1 - relative_path2)

            # Add performance metrics
            logs.append(f"\n📊 Performance Metrics:")
            logs.append(f"├─ Path 1 success: {score_path1:.2%}")
            logs.append(f"├─ Path 2 success: {score_path2:.2%}")
            logs.append(f"├─ Relative path 1 score: {relative_path1:.2%}")
            logs.append(f"├─ Relative path 2 score: {relative_path2:.2%}")
            logs.append(f"└─ Relative difference: {relative_diff:.2%}")
            
            # Add quality metrics for both solutions
            logs.append(f"\n🔍 Solution Quality:")
            logs.append("├─ Path 1:")
            path1_quality = analyze_solution_quality(path1)
            logs.append(f"│  ├─ Length: {path1_quality['length']} words")
            logs.append(f"│  ├─ Steps: {path1_quality['step_count']}")
            logs.append(f"│  └─ Format score: {path1_quality['formatting_quality']}/5")
            
            logs.append("└─ Path 2:")
            path2_quality = analyze_solution_quality(path2)
            logs.append(f"   ├─ Length: {path2_quality['length']} words")
            logs.append(f"   ├─ Steps: {path2_quality['step_count']}")
            logs.append(f"   └─ Format score: {path2_quality['formatting_quality']}/5")
            
            # Check if relative difference is too small (indicating statistical noise)
            if relative_diff < 0.2:
                logs.append("❌ Failed: Score difference too small (< 20%)")
                print("\n".join(logs))
                return {
                    'id': example_id,
                    'status': 'rejected',
                    'reason': 'Score difference too small',
                    'processing_time': time.perf_counter() - start_time
                }
                
            # Swap if path2 has better score
            if score_path2 > score_path1:
                path1, path2 = path2, path1
                score_path1, score_path2 = score_path2, score_path1

            logs.append(f"Score difference: {abs(score_path1 - score_path2)/max(score_path1, score_path2):.1%}")
            
            return {
                'id': example_id,
                'prompt': {'content': bifurcation_prompt, 'role': 'user'},
                'chosen': {'content': path1, 'role': 'assistant'},
                'rejected': {'content': path2, 'role': 'assistant'},
                'score_chosen': score_path1,
                'score_rejected': score_path2,
                'bifurcation_prompt': bifurcation_prompt,
                'quality_metrics': {
                    'chosen': analyze_solution_quality(path1),
                    'rejected': analyze_solution_quality(path2)
                },
                'total_solution_attempts': total_solution_attempts,
                'logs': "\n".join(logs)
            }

        # Print collected logs
        print("\n".join(logs))
            
        # Return consistent format regardless of approach
        processing_time = time.perf_counter() - start_time
        result = {
            'id': example_id,
            'prompt': {'content': bifurcation_prompt, 'role': 'user'},
            'chosen': {'content': path1, 'role': 'assistant'},
            'rejected': {'content': path2, 'role': 'assistant'},
            'score_chosen': score_path1,
            'score_rejected': score_path2,
            'bifurcation_prompt': bifurcation_prompt,
            'quality_metrics': {
                'chosen': analyze_solution_quality(path1),
                'rejected': analyze_solution_quality(path2)
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
            'id': example_id,
            'status': 'error',
            'error_type': type(e).__name__,
            'error_message': str(e),
            'error_category': error_category,
            'processing_time': processing_time,
            'logs': "\n".join(logs)
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
