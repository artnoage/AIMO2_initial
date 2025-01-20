import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.benchmark_utils import *
from utils.agents import *
from utils.log_handler import MarkdownLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

class AdversarialGenerator:
    """Generates pairs of valid correct and incorrect solutions using multiple agents"""
    
    def __init__(self, main, auxiliary, best_of: int, completions: int = 3):
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
                        
            except Exception:
                continue
                
        if step_index == 0:
            self.logs.append(f"Analysis section: Verified={found_verified}, Valid={found_valid}")
        else:
            self.logs.append(f"Step {step_index}: Verified={found_verified}, Valid={found_valid}")
            if not found_verified:
                self.logs.append(f"Step {step_index} is wrong: No verified solutions found")
            elif not found_valid:
                self.logs.append(f"Example dropped: Found verified but no valid solutions at step {step_index}")
                
        return found_verified, found_valid, correct_step, good_completion, completion_prompt

    async def _analyze_wrong_solution(
        self,
        problem: str,
        correct_answer: str,
        wrong_solution: Tuple[str, str],
        correct_solution: str
    ) -> List[Dict[str, Any]]:
        """Analyze wrong solution to find wrong step and generate training examples"""
        results = []
        solution, prompt = wrong_solution
        
        # Split solutions into steps
        wrong_steps = split_into_steps(solution)
        if not wrong_steps or len(wrong_steps) < 2:
            return results
            
        # Get partial solutions
        partial_solutions = get_partial_solutions(wrong_steps)
        num_steps = len(partial_solutions)
        
        # Start with middle step
        current_step = num_steps // 2
        going_up = None
        last_bad_step = None
        last_good_step = None
        wrong_step_index = None
        saved_good_completion = None
        saved_completion_prompt = None
        
        self.logs.append("\n=== Analyzing solution steps ===")
        self.logs.append(f"Starting analysis at step {current_step}")
        
        # Calculate size threshold from correct solution
        size_threshold = int(0.9 * len(correct_solution))
        
        # Binary search for wrong step
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
                
                if found_valid:
                    self.logs.append(f"✓ Step {current_step} is valid")
                    last_good_step = correct_step
                    saved_good_completion = good_completion
                    saved_completion_prompt = completion_prompt
                    
                    if going_up is None:
                        # First check was good, go up to find potential wrong step
                        going_up = True
                    elif not going_up:
                        # We were going down and found a good step
                        # The wrong step must be the last_bad_step we found
                        wrong_step_index = last_bad_step
                        break
                        
                    # Move up one step
                    if current_step + 1 >= num_steps:
                        # We reached the top without finding a wrong step
                        self.logs.append("❌ Reached end without finding wrong step")
                        return []
                    current_step += 1
                    
                else:
                    self.logs.append(f"✗ Step {current_step} cannot be completed correctly")
                    last_bad_step = current_step
                    
                    if going_up is None:
                        # First check was bad, go down to find last good step
                        going_up = False
                        wrong_step_index = current_step  # Save first bad step found
                    elif going_up:
                        # We were going up and found a bad step
                        # The wrong step must be here
                        wrong_step_index = current_step
                        break
                        
                    # Move down one step
                    if current_step - 1 < 0:
                        # We reached the bottom without finding good step
                        self.logs.append("❌ Reached start without finding good step")
                        return []
                    current_step -= 1
                    
            except Exception as e:
                self.logs.append(f"Error in step verification: {str(e)}")
                return []
                
        # If we didn't find a wrong step, return empty results
        if wrong_step_index is None or not saved_good_completion:
            return []
            
        # Create training entries
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
            
            # Add recovery entries
            correct_with_completion = partial_solutions[wrong_step_index-1] + saved_good_completion
            results.append({
                'alignment': 'light',
                'type': 'recovery',
                'problem': problem,
                'prompt': {'content': prompt, 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(correct_with_completion), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(solution), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
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

    async def _run_tournament(
        self,
        solutions: List[Tuple[str, bool, str]],
        problem: str
    ) -> Tuple[List[Tuple[str, bool, str]], List[Dict[str, Any]]]:
        """Run tournament between solutions to rank them and generate training examples"""
        if len(solutions) < 2:
            return solutions

        # Track wins and judge accuracy
        wins = {i: 0 for i in range(len(solutions))}
        judge_correct = 0
        judge_total = 0
        
        # Run round-robin tournament
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                sol_a, is_correct_a, prompt_a = solutions[i]
                sol_b, is_correct_b, prompt_b = solutions[j]
                
                try:
                    # Get judge's decision
                    judge_response = await self.judge_agent.compare_solutions(
                        problem,
                        sol_a,
                        sol_b
                    )
                    
                    # Parse response to get winner (A or B)
                    response = judge_response.strip().upper()
                    if response and response[0] in ['A', 'B']:
                        winner = response[0]
                    else:
                        # Failsafe: randomly choose if response is invalid
                        import random
                        winner = random.choice(['A', 'B'])
                        self.logs.append(f"Invalid judge response, randomly chose {winner}")
                    
                    tournament_results = []
                    # Track judge accuracy when comparing correct vs wrong
                    if is_correct_a != is_correct_b:
                        judge_total += 1
                        if (winner == 'A' and is_correct_a) or (winner == 'B' and is_correct_b):
                            judge_correct += 1
                    
                    # Update wins and check for wrong solution beating correct one
                    winner_idx = i if winner == 'A' else j
                    loser_idx = j if winner == 'A' else i
                    winner_correct = is_correct_a if winner == 'A' else is_correct_b
                    loser_correct = is_correct_b if winner == 'A' else is_correct_a
                    winner_prompt = prompt_a if winner == 'A' else prompt_b
                    winner_sol = sol_a if winner == 'A' else sol_b
                    loser_sol = sol_b if winner == 'A' else sol_a
                    
                    wins[winner_idx] += 1
                    
                    # If wrong solution beat correct solution, add to judge training data
                    if not winner_correct and loser_correct:
                        tournament_results.append({
                            'alignment': 'judge',
                            'type': 'solution',
                            'problem': problem,
                            'prompt': {'content': winner_prompt, 'role': 'user'},
                            'chosen': {'content': remove_inst_tokens(loser_sol), 'role': 'assistant'},
                            'rejected': {'content': remove_inst_tokens(winner_sol), 'role': 'assistant'},
                            'score_chosen': 1.0,
                            'score_rejected': 0.0
                        })
                            
                except Exception as e:
                    self.logs.append(f"Error in tournament match: {str(e)}")
                    continue
        
        # Sort solutions by wins
        sorted_indices = sorted(wins.keys(), key=lambda x: wins[x], reverse=True)
        sorted_solutions = [solutions[i] for i in sorted_indices]
        
        # Create ranking list (1 for correct, 0 for incorrect)
        ranking = [1 if sol[1] else 0 for sol in sorted_solutions]
        
        # Print tournament results
        self.logs.append("\n=== Tournament Results ===")
        self.logs.append(f"Judge accuracy: {judge_correct}/{judge_total} ({judge_correct/judge_total*100:.1f}% correct)")
        self.logs.append(f"Solution ranking (1=correct, 0=incorrect): {ranking}")
        
        return sorted_solutions, tournament_results

    

    async def generate(
        self,
        problem: str,
        correct_answer: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generate both correct and incorrect valid solutions and run tournament.
        Returns list of training examples.
        """
        results = []
        solutions = []
        
        # Search for correct solutions
        attempts = 0
        while attempts < self.best_of:
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
                    self.logs.append(f"✓ Found valid correct solution on attempt {attempts}")
                    
            except Exception as e:
                self.logs.append(f"Error in correct solution attempt {attempts}: {str(e)}")
                continue
                
        # Search for incorrect solutions
        attempts = 0
        while attempts < self.best_of:
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
                    self.logs.append(f"✓ Found valid incorrect solution on attempt {attempts}")
                    
            except Exception as e:
                self.logs.append(f"Error in incorrect solution attempt {attempts}: {str(e)}")
                continue
                
        # Check for at least one correct and one incorrect solution
        has_correct = any(is_correct for _, is_correct, _ in solutions)
        has_incorrect = any(not is_correct for _, is_correct, _ in solutions)
        
        if len(solutions) < 2 or not (has_correct and has_incorrect):
            self.logs.append("Failed to generate required mix of correct and incorrect solutions")
            return None
            
        # Shuffle solutions before tournament
        import random
        random.shuffle(solutions)
        
        # Run tournament to rank solutions
        ranked_solutions, tournament_results = await self._run_tournament(solutions, problem)
        results.extend(tournament_results)
        
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

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig, logger: MarkdownLogger) -> Optional[List[Dict]]:
    """Process a single example using adversarial generation approach"""
    try:
        # Extract answer
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            logging.error(f"Could not extract answer from solution for example {running_id}")
            return None

        # Prepare comprehensive logs for this example
        all_logs = []
        all_logs.append("\n" + "="*80)
        all_logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        all_logs.append("="*80)
        
        # Problem details
        all_logs.append(f"\n📋 Problem:")
        all_logs.append(f"{example['problem'][:200]}...")
        all_logs.append(f"\n✓ Expected Answer: {correct_answer}")

        # Initialize models
        main = get_model(config, role="main")
        auxiliary = get_model(config, role="auxiliary")
        
        # Create generator
        generator = AdversarialGenerator(main, auxiliary, config.best_of)
        
        # Generate solutions and run tournament
        results = await generator.generate(example['problem'], correct_answer)
        if not results:
            return None

        # Add generator logs
        all_logs.extend(generator.logs)
        
        if results:
            # Add solution quality metrics
            all_logs.append("\n📊 Solution Quality:")
            
            # Add details about the generated solutions
            all_logs.append("\n🔍 Generated Solutions:")
            for entry in results:
                all_logs.append(f"✓ Chosen: {entry['chosen']['content'][:200]}...")
                all_logs.append(f"✗ Rejected: {entry['rejected']['content'][:200]}...")
        
        # Print logs for this example
        print("\n".join(all_logs))
        
        # Save comprehensive logs to markdown file
        log_file = logger.save_logs(all_logs, example_id)
        
        # Add example ID to results
        for entry in results:
            entry['id'] = example_id
            
        return results

    except Exception as e:
        logging.error(f"Error processing example {running_id}: {str(e)}")
        return None

async def main():
    """Main function for adversarial generation approach"""
    config = BenchmarkConfig.from_args('Adversarial generation approach')
    logger = MarkdownLogger()  # Create single logger instance for all examples
    
    async def process_with_logger(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[List[Dict]]:
        return await process_example(example, running_id, example_id, config, logger)
    
    await run_benchmark(
        config=config,
        process_example_func=process_with_logger
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
