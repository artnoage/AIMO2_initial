import os
import asyncio
import logging
import re
from typing import Optional, Dict, List, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
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
        size_threshold: int,
    ) -> Tuple[bool, bool, Optional[str], Optional[str], Optional[str]]:
        """Try multiple completions of a partial solution to check if any are correct"""
        found_verified = False
        found_valid = False
        correct_step = None
        good_completion = None
        completion_prompt = None
        self._log(f"\nVerifying completions for step {step_index}:")
        self._log(f"Partial solution length: {len(partial_solution)}")
        for i in range(self.max_attempts):
            try:
                if completion_prompt is None:
                    prompt, completion = await self.completion_agent.generate(
                        problem,
                        partial_solution,
                        return_prompt=True
                    )
                    completion_prompt = prompt
                else:
                    completion = await self.completion_agent.generate(
                        problem,
                        partial_solution
                    )
                complete_solution = partial_solution + completion
                
                # Verify answer correctness
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    found_verified = True
                    self._log(f"✓ Found correct completion on attempt {i+1}")
                        
                    # Check solution size
                    if len(complete_solution) < size_threshold:
                        self._log(f"⚠️ Solution below size threshold: {len(complete_solution)} < {size_threshold}")
                        continue
                        
                    # Validate complete solution
                    is_valid, validation_reason = validate_solution(complete_solution)
                    if is_valid:
                        found_valid = True
                        # Extract next step
                        completion_steps = split_into_steps(complete_solution)
                        if step_index + 1 < len(completion_steps):
                            correct_step = completion_steps[step_index + 1]
                            good_completion = completion
                            self._log(f"✓ Found valid completion with {len(completion_steps)} steps")
                            break
                        else:
                            self._log(f"⚠️ Completion doesn't have enough steps")
                    else:
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
        size_threshold: int = 500
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """Binary search to find first wrong step in solution"""
        # Split solution into steps
        wrong_steps = split_into_steps(wrong_solution)
        if not wrong_steps or len(wrong_steps) < 2:
            self._log("Not enough steps to analyze")
            return None, None, None, None
            
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
        self._log(f"Starting analysis at step {current_step}")
        
        while True:
            try:
                self._log(f"\nChecking step {current_step}...")
                
                found_verified, found_valid, correct_step, good_completion, completion_prompt = await self._verify_completions(
                    problem,
                    partial_solutions[current_step],
                    correct_answer,
                    current_step,
                    size_threshold
                )

                if found_verified and not found_valid:
                    return None, None, None, None
                    
                if found_valid:
                    self._log(f"✓ Step {current_step} is valid")
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
                    self._log(f"✗ Step {current_step} cannot be completed correctly")
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
            # Get steps from wrong solution
            wrong_steps = split_into_steps(wrong_solution)
            
            # Calculate completion score based on remaining steps
            completion_score = wrong_step_index / len(wrong_steps) if len(wrong_steps) > 0 else 0
            
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
                'total_steps': len(wrong_steps)
            })
            
        except Exception as e:
            self._log(f"Error creating training entries: {str(e)}")
            return []

        # Add statistics entry
        stats_result = {
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': True,
            'wrong_step_found': wrong_step_index is not None,
            'wrong_step_index': wrong_step_index if wrong_step_index is not None else -1,
            'total_steps': len(wrong_steps),
            'completion_attempts': self.max_attempts
        }
        
        results.append(stats_result)
        return results

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example to find wrong steps in incorrect solutions"""
    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
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
            logger.print()
            return None

        # Initialize models and agents
        main_model = get_model(config, role="main")
        completion_model = get_model(config, role="auxiliary")
        
        solution_agent = FullSolutionAgent(main_model)
        completion_agent = CompletionAgent(completion_model)
        
        # Create numeric verifier
        verifier = NumericVerifier(tolerance=config.tolerance)
        
        # Generate a solution
        solution = await solution_agent.generate(problem)
        
        # Verify if the solution is correct
        is_correct, model_answer = await verifier.verify(solution, correct_answer, problem)
        
        # Initialize results list
        results = []
        
        # Add the solution to results
        results.append({
            'id': example_id,
            'data_type': 'training',
            'problem': problem,
            'correct_answer': correct_answer,
            'model_solution': solution,
            'model_answer': model_answer,
            'is_correct': is_correct
        })
        
        # If solution is incorrect, analyze to find the wrong step
        if not is_correct:
            logger.append("\n" + "="*80)
            logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
            logger.append("="*80)
            logger.append(f"\n📋 Problem:")
            logger.append(f"{problem[:200]}...")
            logger.append(f"\n✓ Expected Answer: {correct_answer}")
            logger.append(f"\n❌ Solution is incorrect. Model answer: {model_answer}")
            logger.append(f"\n🔍 Analyzing steps to find the error...")
            
            # Create step analyzer with the logger
            analyzer = StepAnalyzer(
                completion_agent=completion_agent,
                solution_agent=solution_agent,
                verifier=verifier,
                max_attempts=config.best_of,
                logs=logger.logs
            )
            
            # Find the wrong step
            wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt = await analyzer.find_wrong_step(
                problem=problem,
                correct_answer=correct_answer,
                wrong_solution=solution
            )
            
            if wrong_step_index is not None:
                logger.append(f"\n✓ Found wrong step at index {wrong_step_index}")
                
                # Get steps from wrong solution
                wrong_steps = split_into_steps(solution)
                partial_solutions = get_partial_solutions(wrong_steps)
                
                # Create training examples
                step_examples = await analyzer.create_step_examples(
                    problem=problem,
                    wrong_solution=solution,
                    correct_answer=correct_answer,
                    wrong_step_index=wrong_step_index,
                    partial_solutions=partial_solutions,
                    saved_good_completion=saved_good_completion,
                    example_id=example_id
                )
                
                # Add step examples to results
                results.extend(step_examples)
            else:
                logger.append(f"\n❌ Could not identify a specific wrong step")
                
                # Add statistics for failed analysis
                results.append({
                    'id': example_id,
                    'data_type': 'statistics',
                    'example_processed_successfully': True,
                    'wrong_step_found': False,
                    'wrong_step_index': -1,
                    'total_steps': len(split_into_steps(solution)),
                    'completion_attempts': config.best_of
                })
        else:
            logger.append("\n" + "="*80)
            logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
            logger.append("="*80)
            logger.append(f"\n📋 Problem:")
            logger.append(f"{problem[:200]}...")
            logger.append(f"\n✓ Expected Answer: {correct_answer}")
            logger.append(f"\n✓ Solution is correct. Model answer: {model_answer}")
            
            # Add statistics for correct solution
            results.append({
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': True,
                'is_correct': True,
                'model_answer': model_answer,
                'total_steps': len(split_into_steps(solution))
            })
            
        # Print all logs at the end
        logger.print()
        
        return results
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        import traceback
        logger.append(f"Traceback:\n{traceback.format_exc()}")
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'is_correct': False,
            'wrong_step_found': False
        }]

async def main():
    """Main function for benchmarking mathematical problem solving with step analysis."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with step analysis')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

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
