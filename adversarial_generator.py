import os
import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.benchmark_utils import *
from utils.agents import *
from utils.log_handler import MarkdownLogger
from utils.tournament_utils import Tournament

# Constants
SIZE_THRESHOLD_FACTOR = 0.9  # Minimum size ratio compared to correct solution


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class AdversarialGenerator:
    """Generates pairs of valid correct and incorrect solutions using multiple agents"""
    
    def __init__(self, main, auxiliary, best_of: int, completions: int):
        self.main = main
        self.auxiliary = auxiliary
        self.best_of = best_of
        self.completions = completions
        self.solution_agent = FullSolutionAgent(main) 
        self.step_agent = NextStepAgent(main)
        self.completion_agent = CompletionAgent(main)
        self.loki_agent = LokiAgent(auxiliary)
        self.judge_agent = TournamentJudgeAgent(auxiliary)
        self.verifier = NumericVerifier()
        self.logs = []
        self.tournament = Tournament(self.judge_agent, logger=self.logs)

    async def _verify_completions(
        self,
        problem: str,
        partial_solution: str,
        correct_answer: str,
        step_index: int,
        size_threshold: int
    ) -> Tuple[bool, bool, Optional[str], Optional[str], Optional[str]]:
        """Try multiple completions of a partial solution to check if any are correct.
        Returns:
            - found_verified: True if any solution verified correctly
            - found_valid: True if any solution both verified and validated
            - correct_step: The next correct step if found, None otherwise
            - good_completion: The full valid completion if found, None otherwise
            - completion_prompt: The prompt used for the successful completion
        """
        found_verified = False
        found_valid = False
        correct_step = None
        good_completion = None
        completion_prompt = None
        
        for i in range(self.completions):
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
                
                # First verify if the answer is correct
                is_correct, _ = await self.verifier.verify(
                    complete_solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    found_verified = True
                    # Validate the completion itself
                    is_valid_completion, completion_reason = validate_completion(partial_solution, completion)
                    if not is_valid_completion:
                        self.logs.append(f"Invalid completion: {completion_reason}")
                        continue
                        
                    # Check solution size
                    if len(complete_solution) < size_threshold:
                        self.logs.append(f"Solution below size threshold: {len(complete_solution)} < {size_threshold}")
                        continue
                        
                    # Then check if the complete solution is valid
                    is_valid, validation_reason = validate_solution(complete_solution)
                    if is_valid:
                        found_valid = True
                        # Extract the next step
                        completion_steps = split_into_steps(complete_solution)
                        next_step_index = step_index + 1
                        correct_step = completion_steps[next_step_index]
                        # Store the successful completion
                        good_completion = completion
                        break
                    else:
                        self.logs.append(f"Found verified but invalid solution: {validation_reason}")
                        continue
                        
            except Exception as e:
                self.logs.append(f"Error in completion attempt {i+1}: {str(e)}")
                continue
                
        if step_index == 0:
            self.logs.append(f"Analysis section: Verified={found_verified}, Valid={found_valid}")
            if not found_verified:
                self.logs.append("Analysis is wrong: No verified solutions found")
            elif not found_valid:
                self.logs.append("Example dropped: Found verified but no valid solutions in analysis")
        else:
            self.logs.append(f"Step {step_index}: Verified={found_verified}, Valid={found_valid}")
            if not found_verified:
                self.logs.append(f"Step {step_index} is wrong: No verified solutions found")
            elif not found_valid:
                self.logs.append(f"Example dropped: Found verified but no valid solutions at step {step_index}")
                
        return found_verified, found_valid, correct_step, good_completion, completion_prompt

    async def _find_wrong_step(
        self,
        problem: str,
        correct_answer: str,
        partial_solutions: List[str],
        size_threshold: int
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """Binary search to find first wrong step in solution"""
        num_steps = len(partial_solutions)
        current_step = num_steps // 2
        going_up = None
        last_bad_step = None
        last_good_step = None
        wrong_step_index = None
        saved_good_completion = None
        saved_completion_prompt = None
        
        self.logs.append("\n=== Analyzing solution steps ===")
        self.logs.append(f"Starting analysis at step {current_step}")
        
        while True:
            try:
                self.logs.append(f"\nChecking step {current_step}...")
                
                found_verified, found_valid, correct_step, good_completion, completion_prompt = await self._verify_completions(
                    problem,
                    partial_solutions[current_step],
                    correct_answer,
                    current_step,
                    size_threshold
                )

                if found_verified and not found_valid:
                    self.logs.append("Found verified but invalid completion - skipping step analysis")
                    return None, None, None, None
                    
                if found_valid:
                    self.logs.append(f"✓ Step {current_step} is valid")
                    last_good_step = correct_step
                    saved_good_completion = good_completion
                    saved_completion_prompt = completion_prompt
                    
                    if going_up is None:
                        going_up = True
                    elif not going_up:
                        wrong_step_index = last_bad_step
                        break
                        
                    if current_step + 1 >= num_steps:
                        self.logs.append("❌ Reached end without finding wrong step")
                        return None, None, None, None
                    current_step += 1
                    
                else:
                    self.logs.append(f"✗ Step {current_step} cannot be completed correctly")
                    last_bad_step = current_step
                    
                    if going_up is None:
                        going_up = False
                        wrong_step_index = current_step
                    elif going_up:
                        wrong_step_index = current_step
                        break
                        
                    if current_step - 1 < 0:
                        self.logs.append("❌ Reached start without finding good step")
                        return None, None, None, None
                    current_step -= 1
                    
            except Exception as e:
                self.logs.append(f"Error in step verification: {str(e)}")
                return None, None, None, None
                
        return wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt

    async def _create_step_examples(
        self,
        problem: str,
        wrong_solution: Tuple[str, str],
        wrong_steps: List[str],
        partial_solutions: List[str],
        wrong_step_index: int,
        last_good_step: str,
        saved_good_completion: str,
        saved_completion_prompt: str
    ) -> List[Dict[str, Any]]:
        """Create training examples from identified wrong step"""
        results = []
        solution, prompt = wrong_solution
        
        try:
            # Get step prompt
            step_prompt = await self.step_agent.generate(
                problem,
                partial_solutions[max(0, wrong_step_index - 1)],
                return_prompt=True
            )
            
            # Add step entry
            results.append({
                'alignment': 'light',
                'type': 'step',
                'problem': problem,
                'prompt': {'content': step_prompt[0], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(last_good_step), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(wrong_steps[wrong_step_index]), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
            # Add completion entry
            results.append({
                'alignment': 'light', 
                'type': 'completion',
                'problem': problem,
                'prompt': {'content': saved_completion_prompt, 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(saved_good_completion), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(''.join(wrong_steps[wrong_step_index:])), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
            # Get solver prompt for light recovery
            solver_prompt = await self.solution_agent.generate(problem, return_prompt=True)
            
            # Add recovery entries
            correct_with_completion = partial_solutions[wrong_step_index-1] + saved_good_completion
            results.append({
                'alignment': 'light',
                'type': 'recovery',
                'problem': problem,
                'prompt': {'content': solver_prompt[0], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(correct_with_completion), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(solution), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
            # Use Loki prompt for dark recovery
            results.append({
                'alignment': 'dark',
                'type': 'recovery', 
                'problem': problem,
                'prompt': {'content': prompt, 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(solution), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(correct_with_completion), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
        except Exception as e:
            self.logs.append(f"Error creating training entries: {str(e)}")
            return []
            
        return results

    async def _analyze_wrong_solution(
        self,
        problem: str,
        correct_answer: str,
        wrong_solution: Tuple[str, str],
        correct_solution: str
    ) -> List[Dict[str, Any]]:
        """Analyze wrong solution to find wrong step and generate training examples"""
        solution, _ = wrong_solution
        
        # Split solutions into steps
        wrong_steps = split_into_steps(solution)
        if not wrong_steps or len(wrong_steps) < 2:
            return []
            
        # Get partial solutions
        partial_solutions = get_partial_solutions(wrong_steps)
        size_threshold = int(SIZE_THRESHOLD_FACTOR * len(correct_solution))
        
        # Find wrong step
        wrong_step_index, last_good_step, saved_good_completion, saved_completion_prompt = await self._find_wrong_step(
            problem,
            correct_answer,
            partial_solutions,
            size_threshold
        )
        
        # If we didn't find a wrong step, return empty results
        if wrong_step_index is None or not saved_good_completion:
            return []
            
        # Create training examples
        return await self._create_step_examples(
            problem,
            wrong_solution,
            wrong_steps,
            partial_solutions,
            wrong_step_index,
            last_good_step,
            saved_good_completion,
            saved_completion_prompt
        )

    async def _generate_solutions(
        self,
        problem: str,
        correct_answer: str
    ) -> List[Tuple[str, bool, str]]:
        """Generate both correct and incorrect solutions"""
        solutions = []
        correct_count = 0
        incorrect_count = 0
        
        # Search for correct solutions first
        attempts = 0
        while attempts < self.best_of and correct_count < 3:
            try:
                attempts += 1
                prompt, solution = await self.solution_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logs.append(f"Invalid solution structure: {validation_reason}")
                    continue
                    
                # Verify correctness
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if is_correct:
                    solutions.append((solution, True, prompt))
                    correct_count += 1
                    self.logs.append(f"✓ Found valid correct solution on attempt {attempts} ({correct_count}/3)")
                    
            except Exception as e:
                self.logs.append(f"Error in correct solution attempt {attempts}: {str(e)}")
                continue

        # Only search for incorrect solutions if we found at least one correct solution
        if correct_count == 0:
            self.logs.append("Failed to find any correct solutions - skipping incorrect solution generation")
            return solutions

        # Search for incorrect solutions
        attempts = 0
        while attempts < self.best_of and incorrect_count < 3:
            try:
                attempts += 1
                prompt, solution = await self.loki_agent.generate(problem, return_prompt=True)
                
                # Validate solution structure
                is_valid, validation_reason = validate_solution(solution)
                if not is_valid:
                    self.logs.append(f"Invalid Loki solution structure: {validation_reason}")
                    continue
                    
                # Verify incorrectness
                is_correct, _ = await self.verifier.verify(
                    solution,
                    correct_answer,
                    problem
                )
                
                if not is_correct:
                    solutions.append((solution, False, prompt))
                    incorrect_count += 1
                    self.logs.append(f"✓ Found valid incorrect solution on attempt {attempts} ({incorrect_count}/3)")
                    
            except Exception as e:
                self.logs.append(f"Error in incorrect solution attempt {attempts}: {str(e)}")
                continue
                
        return solutions

    async def _create_training_examples(
        self,
        problem: str,
        correct_answer: str,
        ranked_solutions: List[Tuple[str, bool, str]]
    ) -> List[Dict[str, Any]]:
        """Create training examples from ranked solutions"""
        results = []
        
        # Find top solutions
        top_correct = None
        top_wrong = None
        second_wrong = None
        for solution, is_correct, prompt in ranked_solutions:
            if is_correct and top_correct is None:
                top_correct = (solution, prompt)
            elif not is_correct:
                if top_wrong is None:
                    top_wrong = (solution, prompt)
                elif second_wrong is None:
                    second_wrong = (solution, prompt)
                    break
                    
        # Process top wrong solution for step-based entries
        if top_wrong and top_correct:
            step_results = await self._analyze_wrong_solution(problem, correct_answer, top_wrong, top_correct[0])
            results.extend(step_results)
        
        # Create training examples
        if top_correct and top_wrong:
            # Light alignment example
            results.append({
                'alignment': 'light',
                'type': 'full_solution',
                'problem': problem,
                'prompt': {'content': top_correct[1], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(top_correct[0]), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(top_wrong[0]), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
        # Create dark entry if we have two wrong solutions
        if top_wrong and second_wrong:
            results.append({
                'alignment': 'dark',
                'type': 'full_solution',
                'problem': problem,
                'prompt': {'content': top_wrong[1], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(top_wrong[0]), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(second_wrong[0]), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
        return results

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> List[Dict[str, Any]]:
        """
        Generate both correct and incorrect valid solutions and run tournament.
        Returns list of training examples.
        """
        # Generate solutions
        solutions = await self._generate_solutions(problem, correct_answer)
        
        # Check for at least one correct and one incorrect solution
        has_correct = any(is_correct for _, is_correct, _ in solutions)
        has_incorrect = any(not is_correct for _, is_correct, _ in solutions)
        
        if len(solutions) < 2 or not (has_correct and has_incorrect):
            self.logs.append("Failed to generate required mix of correct and incorrect solutions")
            return []
            
        # Shuffle solutions before tournament
        random.shuffle(solutions)
        
        # Run tournament to rank solutions
        ranked_solutions, tournament_results, _ = await self.tournament.run_tournament(solutions, problem)
        
        # Create training examples
        results = await self._create_training_examples(problem, correct_answer, ranked_solutions)
        results.extend(tournament_results)
        
        return results

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig, logger: MarkdownLogger) -> List[Dict]:
    """Process a single example using adversarial generation approach"""
    try:
        # Extract answer
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            logging.error(f"Could not extract answer from solution for example {running_id}")
            return []

        # Initialize models
        main = get_model(config, role="main")
        auxiliary = get_model(config, role="auxiliary")
        
        # Create generator
        generator = AdversarialGenerator(main, auxiliary, config.best_of, config.completions)
        
        # Prepare logs
        all_logs = []
        all_logs.append("\n" + "="*80)
        all_logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        all_logs.append("="*80)
        all_logs.append(f"\n📋 Problem:")
        all_logs.append(f"{example['problem'][:200]}...")
        all_logs.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Generate solutions and run tournament
        results = await generator.generate(example['problem'], correct_answer)
        
        # Add example ID to results
        if results:
            for entry in results:
                entry['id'] = example_id
                
        # Add generator logs and solution details
        all_logs.extend(generator.logs)
        if results:
            all_logs.append("\n📊 Solution Quality:")
            all_logs.append("\n🔍 Generated Solutions:")
            for entry in results:
                all_logs.append(f"✓ Chosen: {entry['chosen']['content'][:200]}...")
                all_logs.append(f"✗ Rejected: {entry['rejected']['content'][:200]}...")
        
        # Print and save logs
        print("\n".join(all_logs))
        logger.save_logs(all_logs, example_id)
            
        return results

    except Exception as e:
        logging.error(f"Error processing example {running_id}: {str(e)}")
        return []

async def main():
    """Main function for adversarial generation approach"""
    config = BenchmarkConfig.from_args('Adversarial generation approach')
    logger = MarkdownLogger()  # Create single logger instance for all examples
    
    await run_benchmark(
        config=config,
        process_example_func=lambda example, running_id, example_id, config: process_example(
            example, running_id, example_id, config, logger
        )
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
