import os
import re
import asyncio
import logging
from typing import Optional, Dict, List, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.model_utils import *
from utils.solution_utils import *
from utils.agents import *
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class StepAnalyzer:
    """Analyzes solutions to find wrong steps and generate training examples"""
    
    def __init__(self, completion_agent, solution_agent, verifier, max_attempts=5, logs=None):
        """
        Initialize step analyzer
        Args:
            completion_agent: Agent for completing partial solutions
            solution_agent: Agent for full solutions
            verifier: Numeric answer verifier
            max_attempts: Maximum number of completion attempts
            logs: Optional list for logging
        """
        self.completion_agent = completion_agent
        self.solution_agent = solution_agent
        self.verifier = verifier
        self.max_attempts = max_attempts
        self.logs = logs if logs is not None else []

    def _log(self, message: str):
        """Add message to logs if logging is enabled"""
        if self.logs is not None:
            # Add a prefix to step analyzer logs for better visibility
            self.logs.append(f"[Step Analysis] {message}")

    async def _verify_completions(
        self,
        problem: str,
        partial_solution: str,
        correct_answer: str,
        step_index: int,
        num_completions: int,
    ) -> Tuple[bool, bool, Optional[str], Optional[str], Optional[str]]:
        """Try multiple completions of a partial solution to check if any are correct"""
        found_verified = False
        found_valid = False
        correct_step = None
        good_completion = None
        completion_prompt = None
        self._log(f"\nVerifying completions for step {step_index+1} (index {step_index}):")
        self._log(f"Partial solution length: {len(partial_solution)}")
        self._log(f"Trying {num_completions} completions")
        
        # We're always using <step> tags format in the response section
        self._log(f"Using <step> tags format")
        
        # Generate all completions at once
        completions = []
        for i in range(num_completions):
            try:
                if i == 0 and completion_prompt is None:
                    prompt, completion = await self.completion_agent.generate(
                        problem,
                        partial_solution,
                        return_prompt=True
                    )
                    completion_prompt = prompt
                    completions.append(completion)
                else:
                    completion = await self.completion_agent.generate(
                        problem,
                        partial_solution
                    )
                    completions.append(completion)
            except Exception as e:
                self._log(f"Error generating completion {i+1}: {str(e)}")
                
        # Evaluate all completions
        for i, completion in enumerate(completions):
            try:
                # Check if completion has proper format with <step> tags
                if not ("<step>" in completion):
                    self._log(f"⚠️ Completion {i+1} missing <step> tags, skipping")
                    continue
                
                # Extract content from response tags if present
                completion_content = extract_response_section(completion) or completion
                complete_solution = partial_solution + completion_content
                
                # Verify answer correctness
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    found_verified = True
                    self._log(f"✓ Found correct completion on attempt {i+1}")
                        
                    # Validate complete solution
                    is_valid = True
                    validation_reason = "Valid solution"
                    
                    # Extract next step
                    completion_steps = split_into_steps(complete_solution)
                    if step_index + 1 < len(completion_steps):
                        correct_step = completion_steps[step_index + 1]
                        good_completion = completion
                        self._log(f"✓ Found valid completion with {len(completion_steps)} steps")
                        break
                    else:
                        self._log(f"⚠️ Completion doesn't have enough steps")
                        is_valid = False
                        validation_reason = "Not enough steps"
                        
                    if not is_valid:
                        self._log(f"Found verified but invalid solution: {validation_reason}")
                        continue
                        
            except Exception as e:
                self._log(f"Error in completion attempt {i+1}: {str(e)}")
                continue
                
        # Log verification results
        if step_index == 0:
            self._log(f"Analysis section: Verified={found_verified}, Valid={found_valid}")
        else:
            self._log(f"Step {step_index}: Verified={found_verified}, Valid={found_valid}")
            
        return found_verified, found_valid, correct_step, good_completion, completion_prompt

    async def find_wrong_step(
        self,
        problem: str,
        correct_answer: str,
        wrong_solution: str,
        num_completions: int = 10
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """Binary search to find first wrong step in solution"""
        # Split solution into steps - these should already be just the content inside <step> tags
        wrong_steps = split_into_steps(wrong_solution)
        if not wrong_steps or len(wrong_steps) < 2:
            self._log("Not enough steps to analyze")
            return None, None, None, None
        
        # We're always using <step> tags format in the response section
        self._log(f"Using <step> tags format")
        
        # Count steps
        step_count = len(wrong_steps)
        self._log(f"Found {step_count} steps in solution")
        
        # Get partial solutions
        partial_solutions = get_partial_solutions(wrong_steps)
        num_steps = len(partial_solutions)
        
        # Binary search variables
        current_step = num_steps // 2
        going_up = None
        last_bad_step = None
        last_good_step = None
        wrong_step_index = None
        saved_good_completion = None
        saved_completion_prompt = None
        
        self._log("\n=== Analyzing solution steps ===")
        self._log(f"Starting analysis at step {current_step+1} (index {current_step})")
        
        while True:
            try:
                self._log(f"\nChecking step {current_step}...")
                
                found_verified, found_valid, correct_step, good_completion, completion_prompt = await self._verify_completions(
                    problem,
                    partial_solutions[current_step],
                    correct_answer,
                    current_step,
                    num_completions
                )

                if found_verified and not found_valid:
                    self._log(f"✗ Found verified but invalid solution at step {current_step+1}")
                    return None, None, None, None
                    
                if found_valid:
                    self._log(f"✓ Step {current_step+1} (index {current_step}) is valid")
                    last_good_step = correct_step
                    saved_good_completion = good_completion
                    saved_completion_prompt = completion_prompt
                    
                    if going_up is None:
                        going_up = True
                    elif not going_up:
                        wrong_step_index = last_bad_step
                        break
                        
                    if current_step + 1 >= num_steps:
                        self._log("❌ Reached end without finding wrong step")
                        return None, None, None, None
                    current_step += 1
                    
                else:
                    self._log(f"✗ Step {current_step+1} (index {current_step}) cannot be completed correctly")
                    last_bad_step = current_step
                    
                    if going_up is None:
                        going_up = False
                        wrong_step_index = current_step
                    elif going_up:
                        wrong_step_index = current_step
                        break
                        
                    if current_step - 1 < 0:
                        self._log("❌ Reached start without finding good step")
                        return None, None, None, None
                    current_step -= 1
                    
            except Exception as e:
                self._log(f"Error in step verification: {str(e)}")
                return None, None, None, None
                
        return wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt

    async def create_step_examples(
        self,
        problem: str,
        wrong_solution: str,
        correct_answer: str,
        wrong_step_index: int,
        partial_solutions: List[str],
        saved_good_completion: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Create training examples from identified wrong step"""
        results = []
        
        try:
            # Get steps from wrong solution - these should already be just the content inside <step> tags
            wrong_steps = split_into_steps(wrong_solution)
            
            # Calculate completion score based on remaining steps
            completion_score = wrong_step_index / len(wrong_steps) if len(wrong_steps) > 0 else 0
            
            # Calculate position category (beginning, middle, end)
            if len(wrong_steps) <= 2:
                position_category = "short_solution"
            elif wrong_step_index == 0:
                position_category = "beginning"  # First step (Step 1)
            elif wrong_step_index == len(wrong_steps) - 1:
                position_category = "end"  # Last step
            elif wrong_step_index < len(wrong_steps) / 3:
                position_category = "early"
            elif wrong_step_index < 2 * len(wrong_steps) / 3:
                position_category = "middle"
            else:
                position_category = "late"
            
            # Get solver prompt for recovery
            solver_prompt = await self.solution_agent.generate(problem, return_prompt=True)
            
            # Create correct solution with completion
            correct_with_completion = partial_solutions[wrong_step_index-1] + saved_good_completion if wrong_step_index > 0 else saved_good_completion
            
            # Add recovery entry (training data)
            results.append({
                'data_type': 'training',
                'type': 'recovery',
                'problem': problem,
                'correct_answer': correct_answer,
                'wrong_solution': wrong_solution,
                'corrected_solution': correct_with_completion,
                'wrong_step_index': wrong_step_index,
                'total_steps': len(wrong_steps),
                'position_category': position_category,
                'completion_score': completion_score
            })
            
        except Exception as e:
            self._log(f"Error creating training entries: {str(e)}")
            return []

        # Add statistics entry with enhanced metrics
        stats_result = {
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': True,
            'wrong_step_found': wrong_step_index is not None,
            'wrong_step_index': wrong_step_index if wrong_step_index is not None else -1,
            'total_steps': len(wrong_steps),
            'completion_attempts': self.max_attempts,
            'num_completions_per_step': self.max_attempts,
            'position_category': position_category,
            'completion_score': completion_score,
            'solution_length': len(wrong_solution),
            'wrong_step_relative_position': wrong_step_index / len(wrong_steps) if len(wrong_steps) > 0 else 0
        }
        
        results.append(stats_result)
        return results

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example to find wrong steps in incorrect solutions"""
    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            return None
            
        # Extract problem and correct answer
        problem = example['problem']
        correct_answer = None
        
        # If example has a solution, extract answer from it
        if 'solution' in example:
            correct_answer = extract_answer_from_solution(example['solution'])
        # Otherwise use the answer field directly
        elif 'answer' in example:
            correct_answer = example['answer']
            
        if correct_answer is None:
            logger.append(f"❌ Warning: Could not determine correct answer for example {str(running_id)}")
            return None

        # Initialize models and agents - all using main model
        main_model = get_model(config, role="main")
        
        solution_agent = FullSolutionAgent(main_model)
        completion_agent = CompletionAgent(main_model)
        
        # Create numeric verifier
        verifier = NumericVerifier(tolerance=config.tolerance)
        
        # Use extract functions from benchmark_utils
        
        # Generate solutions (always generate best_of solutions)
        solutions = []
        for i in range(config.best_of):
            solution = await solution_agent.generate(problem)
            is_correct, model_answer = await verifier.verify(solution, correct_answer, problem)
            
            # Extract thinking and response sections if present
            thinking = extract_thinking_section(solution)
            response = extract_response_section(solution)
            
            solutions.append({
                'solution': solution,
                'answer': model_answer,
                'is_correct': is_correct,
                'thinking': thinking,
                'response': response
            })
        
        # Initialize results list
        results = []
        
        # Add all solutions to results
        for i, sol_data in enumerate(solutions):
            results.append({
                'id': example_id,
                'data_type': 'training',
                'problem': problem,
                'correct_answer': correct_answer,
                'model_solution': sol_data['solution'],
                'model_answer': sol_data['answer'],
                'is_correct': sol_data['is_correct'],
                'attempt_number': i + 1,
                'total_attempts': len(solutions)
            })
        
        # Analyze each incorrect solution to find wrong steps
        incorrect_solutions = [s for s in solutions if not s['is_correct']]
        
        # If we have any incorrect solutions, analyze them
        if incorrect_solutions:
            logger.append("\n" + "="*80)
            logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
            logger.append("="*80)
            logger.append(f"\n📋 Problem:")
            logger.append(f"{problem[:200]}...")
            logger.append(f"\n✓ Expected Answer: {correct_answer}")
            logger.append(f"\n❌ Found {len(incorrect_solutions)} incorrect solutions")
            logger.append(f"\n🔍 Analyzing steps to find errors...")
            
            # Create step analyzer with the logger
            analyzer = StepAnalyzer(
                completion_agent=completion_agent,
                solution_agent=solution_agent,
                verifier=verifier,
                max_attempts=config.completions,  # Use the completions parameter
                logs=logger.logs
            )
            
            # Analyze each incorrect solution (up to 3 to avoid excessive processing)
            analyzed_solutions = []
            for idx, sol_data in enumerate(incorrect_solutions[:3]):
                solution = sol_data['solution']
                model_answer = sol_data['answer']
                thinking = sol_data.get('thinking')
                response = sol_data.get('response')
                
                logger.append(f"\n🔍 Analyzing incorrect solution {idx+1}/{min(3, len(incorrect_solutions))}")
                logger.append(f"   Model answer: {model_answer}")
                
                # If we have thinking/response sections, log them
                if thinking:
                    logger.append(f"\n📝 Thinking section extracted:")
                    logger.append(f"{thinking[:200]}...")
                
                if response:
                    logger.append(f"\n📝 Response section extracted:")
                    logger.append(f"{response[:200]}...")
                
                # Check if response section exists
                if not response or len(response.strip()) == 0:
                    logger.append(f"   ❌ No response section found - marking as unsalvageable")
                    analyzed_solutions.append({
                        'solution': solution,
                        'wrong_step_index': None,
                        'good_completion': None,
                        'last_good_step': None,
                        'unsalvageable': True,
                        'reason': 'no_response_section'
                    })
                    continue
                
                # Use response section for analysis
                logger.append(f"   Using extracted response section for analysis")
                analysis_solution = response
                
                # Check if response contains steps with proper tags and has enough steps
                steps = split_into_steps(response)
                
                if not steps or '<step>' not in response:
                    logger.append(f"   ❌ No <step> tags found in response section - marking as unsalvageable")
                    analyzed_solutions.append({
                        'solution': solution,
                        'wrong_step_index': None,
                        'good_completion': None,
                        'last_good_step': None,
                        'unsalvageable': True,
                        'reason': 'no_step_tags_in_response'
                    })
                    continue
                
                if len(steps) < 2:
                    logger.append(f"   ❌ Not enough steps in response (found {len(steps)}) - marking as unsalvageable")
                    analyzed_solutions.append({
                        'solution': solution,
                        'wrong_step_index': None,
                        'good_completion': None,
                        'last_good_step': None,
                        'unsalvageable': True,
                        'reason': 'insufficient_steps'
                    })
                    continue
                
                # Extract and validate step numbers
                step_numbers = []
                valid_step_numbers = True
                
                for i, step in enumerate(steps):
                    step_num = None
                    for pattern in STEP_NUMBER_PATTERNS:
                        match = pattern.search(step)
                        if match:
                            try:
                                step_num = int(match.group(1))
                                break
                            except ValueError:
                                continue
                    
                    if step_num is None:
                        logger.append(f"   ❌ Step {i+1} has no valid step number - marking as unsalvageable")
                        valid_step_numbers = False
                        break
                    
                    step_numbers.append(step_num)
                
                if not valid_step_numbers:
                    analyzed_solutions.append({
                        'solution': solution,
                        'wrong_step_index': None,
                        'good_completion': None,
                        'last_good_step': None,
                        'unsalvageable': True,
                        'reason': 'missing_step_numbers'
                    })
                    continue
                
                # Check if step numbers are sequential
                expected_numbers = list(range(1, len(steps) + 1))
                if step_numbers != expected_numbers:
                    logger.append(f"   ❌ Step numbers are not sequential: {step_numbers} - marking as unsalvageable")
                    analyzed_solutions.append({
                        'solution': solution,
                        'wrong_step_index': None,
                        'good_completion': None,
                        'last_good_step': None,
                        'unsalvageable': True,
                        'reason': 'non_sequential_step_numbers'
                    })
                    continue
                
                logger.append(f"   Found {len(steps)} steps in response")
                
                # Find the wrong step
                wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt = await analyzer.find_wrong_step(
                    problem=problem,
                    correct_answer=correct_answer,
                    wrong_solution=analysis_solution,
                    num_completions=config.completions
                )
                
                analyzed_solutions.append({
                    'solution': solution,
                    'wrong_step_index': wrong_step_index,
                    'good_completion': saved_good_completion,
                    'last_good_step': last_good_step
                })
            
            # Process all analyzed solutions
            for idx, analysis in enumerate(analyzed_solutions):
                solution = analysis['solution']
                wrong_step_index = analysis['wrong_step_index']
                saved_good_completion = analysis['good_completion']
                
                if 'unsalvageable' in analysis and analysis['unsalvageable']:
                    logger.append(f"\n❌ Solution {idx+1} is unsalvageable: {analysis['reason']}")
                    
                    # Add statistics for unsalvageable solution
                    results.append({
                        'id': example_id,
                        'data_type': 'statistics',
                        'example_processed_successfully': True,
                        'wrong_step_found': False,
                        'wrong_step_index': -1,
                        'total_steps': 0,
                        'completion_attempts': 0,
                        'solution_index': idx,
                        'unsalvageable': True,
                        'unsalvageable_reason': analysis['reason']
                    })
                elif wrong_step_index is not None:
                    logger.append(f"\n✓ Found wrong step {wrong_step_index+1} (index {wrong_step_index}) in solution {idx+1}")
                    
                    # Get steps from wrong solution
                    wrong_steps = split_into_steps(analysis_solution)
                    partial_solutions = get_partial_solutions(wrong_steps)
                    
                    # Create training examples
                    step_examples = await analyzer.create_step_examples(
                        problem=problem,
                        wrong_solution=analysis_solution,
                        correct_answer=correct_answer,
                        wrong_step_index=wrong_step_index,
                        partial_solutions=partial_solutions,
                        saved_good_completion=saved_good_completion,
                        example_id=example_id
                    )
                    
                    # Add the original solution and reasoning to the first training example
                    if step_examples and len(step_examples) > 0:
                        for example in step_examples:
                            if example.get('data_type') == 'training':
                                example['original_solution'] = solution
                                if thinking:
                                    example['thinking'] = thinking
                                if response:
                                    example['response'] = response
                                break
                    
                    # Add step examples to results
                    results.extend(step_examples)
                else:
                    logger.append(f"\n❌ Could not identify a specific wrong step in solution {idx+1}")
                    
                    # Add statistics for failed analysis
                    results.append({
                        'id': example_id,
                        'data_type': 'statistics',
                        'example_processed_successfully': True,
                        'wrong_step_found': False,
                        'wrong_step_index': -1,
                        'total_steps': len(split_into_steps(analysis_solution)),
                        'completion_attempts': config.completions,
                        'solution_index': idx,
                        'unsalvageable': False
                    })
            
            # Calculate step benchmark specific statistics
            # Collect statistics about wrong steps
            wrong_step_positions = []
            position_categories = []
            completion_scores = []
            
            for result in results:
                if result.get('data_type') == 'statistics' and result.get('wrong_step_found', False):
                    if 'wrong_step_relative_position' in result:
                        wrong_step_positions.append(result['wrong_step_relative_position'])
                    if 'position_category' in result:
                        position_categories.append(result['position_category'])
                    if 'completion_score' in result:
                        completion_scores.append(result['completion_score'])
            
            # Count position categories
            from collections import Counter
            position_counts = Counter(position_categories)
            
            # Calculate average position and completion score
            avg_position = sum(wrong_step_positions) / len(wrong_step_positions) if wrong_step_positions else 0
            avg_completion_score = sum(completion_scores) / len(completion_scores) if completion_scores else 0
            
            # Count solutions with thinking/response sections
            solutions_with_thinking = sum(1 for s in incorrect_solutions[:3] if s.get('thinking'))
            solutions_with_response = sum(1 for s in incorrect_solutions[:3] if s.get('response'))
            
            # Count unsalvageable solutions
            unsalvageable_solutions = sum(1 for r in results if r.get('data_type') == 'statistics' and r.get('unsalvageable', False))
            unsalvageable_reasons = {}
            for r in results:
                if r.get('data_type') == 'statistics' and r.get('unsalvageable', False):
                    reason = r.get('unsalvageable_reason', 'unknown')
                    unsalvageable_reasons[reason] = unsalvageable_reasons.get(reason, 0) + 1
            
            # Update the statistics entry with step-specific information
            # Find the existing statistics entry or create a new one
            stats_entry = next((r for r in results if r.get('data_type') == 'statistics'), None)
            
            if stats_entry:
                # Update existing statistics entry
                stats_entry.update({
                    'wrong_steps_found': len(wrong_step_positions),
                    'avg_wrong_step_position': avg_position,
                    'position_distribution': dict(position_counts),
                    'avg_completion_score': avg_completion_score,
                    'recovery_success_rate': len(wrong_step_positions) / len(incorrect_solutions[:3]) if incorrect_solutions else 0,
                    'solutions_with_thinking': solutions_with_thinking,
                    'solutions_with_response': solutions_with_response,
                    'thinking_extraction_rate': solutions_with_thinking / len(incorrect_solutions[:3]) if incorrect_solutions[:3] else 0,
                    'response_extraction_rate': solutions_with_response / len(incorrect_solutions[:3]) if incorrect_solutions[:3] else 0,
                    'unsalvageable_solutions': unsalvageable_solutions,
                    'unsalvageable_rate': unsalvageable_solutions / len(incorrect_solutions[:3]) if incorrect_solutions[:3] else 0,
                    'unsalvageable_reasons': unsalvageable_reasons
                })
            else:
                # Create a new statistics entry
                results.append({
                    'id': example_id,
                    'data_type': 'statistics',
                    'example_processed_successfully': True,
                    'wrong_steps_found': len(wrong_step_positions),
                    'avg_wrong_step_position': avg_position,
                    'position_distribution': dict(position_counts),
                    'avg_completion_score': avg_completion_score,
                    'recovery_success_rate': len(wrong_step_positions) / len(incorrect_solutions[:3]) if incorrect_solutions else 0,
                    'solutions_with_thinking': solutions_with_thinking,
                    'solutions_with_response': solutions_with_response,
                    'thinking_extraction_rate': solutions_with_thinking / len(incorrect_solutions[:3]) if incorrect_solutions[:3] else 0,
                    'response_extraction_rate': solutions_with_response / len(incorrect_solutions[:3]) if incorrect_solutions[:3] else 0,
                    'unsalvageable_solutions': unsalvageable_solutions,
                    'unsalvageable_rate': unsalvageable_solutions / len(incorrect_solutions[:3]) if incorrect_solutions[:3] else 0,
                    'unsalvageable_reasons': unsalvageable_reasons
                })
        else:
            logger.append("\n" + "="*80)
            logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
            logger.append("="*80)
            logger.append(f"\n📋 Problem:")
            logger.append(f"{problem[:200]}...")
            logger.append(f"\n✓ Expected Answer: {correct_answer}")
            
            # Log all solution results
            for i, sol_data in enumerate(solutions):
                logger.append(f"\n{'✓' if sol_data['is_correct'] else '❌'} Solution {i+1}: {sol_data['answer']}")
            
            # Calculate statistics
            correct_count = sum(1 for s in solutions if s['is_correct'])
            success_rate = (correct_count / len(solutions)) * 100 if solutions else 0
            
            logger.append(f"\n📊 Statistics:")
            logger.append(f"├─ Correct solutions: {correct_count}/{len(solutions)}")
            logger.append(f"└─ Success rate: {success_rate:.1f}%")
            
            # Add statistics for all solutions
            results.append({
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': True,
                'is_correct_list': [s['is_correct'] for s in solutions],
                'success_rate': success_rate,
                'total_solutions': len(solutions),
                'correct_solutions': correct_count,
                'incorrect_solutions': len(solutions) - correct_count,
                'total_steps': [len(split_into_steps(s['solution'])) for s in solutions]
            })
            
        # Print logs before returning results
        logger.print()
        
        # Return results with logs
        return {
            'results': results,
            'logs': '\n'.join(logger.logs) if logger.logs else ""
        }
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        import traceback
        logger.append(f"Traceback:\n{traceback.format_exc()}")
        
        # Print logs before returning results
        logger.print()
        
        return {
            'results': [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'is_correct': False,
                'wrong_step_found': False
            }],
            'logs': '\n'.join(logger.logs) if logger.logs else ""
        }

async def main():
    """Main function for benchmarking mathematical problem solving with step analysis."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with step analysis')
    
    # Create a logger for collecting all logs
    all_logs = BenchmarkLogger()
    
    # Run the benchmark
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)
    
    # No need to print all logs at the end since we're printing them with each entry
    
    # The ProgressTracker will handle printing the final statistics

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        logger.append(f"Traceback:\n{traceback.format_exc()}")
        logger.print()
