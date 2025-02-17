import random
from typing import Dict, List, Optional, Tuple, Any
from utils.tournament_utils import Tournament
from utils.benchmark_utils import (
    validate_solution,
    split_into_steps,
    get_partial_solutions,
    remove_inst_tokens
)

class StepAnalyzer:
    """Analyzes solutions to find wrong steps and generate training examples"""
    
    def __init__(self, completion_agent, solution_agent, verifier, max_attempts, logs=None):
        """
        Initialize step analyzer
        Args:
            completion_agent: Agent for completing partial solutions
            step_agent: Agent for generating next steps
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
                        next_step_index = step_index + 1
                        correct_step = completion_steps[next_step_index]
                        good_completion = completion
                        self._log(f"✓ Found valid completion with {len(completion_steps)} steps")
                        break
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
        size_threshold: int
    ) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        """Binary search to find first wrong step in solution"""
        # Split solution into steps
        wrong_steps = split_into_steps(wrong_solution)
        if not wrong_steps or len(wrong_steps) < 2:
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
        wrong_solution: Tuple[str, str],
        wrong_steps: List[str],
        partial_solutions: List[str],
        wrong_step_index: int,
        saved_good_completion: str,
        saved_completion_prompt: str,
        example_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Create training examples from identified wrong step"""
        results = []
        solution, prompt = wrong_solution
        
        try:
            # Calculate completion score based on remaining steps
            completion_score = wrong_step_index / len(wrong_steps)
            
            # Get solver prompt for recovery
            solver_prompt = await self.solution_agent.generate(problem, return_prompt=True)
            
            # Add recovery entry (light alignment)
            correct_with_completion = partial_solutions[wrong_step_index-1] + saved_good_completion
            results.append({
                'data_type': 'training',
                'alignment': 'light',
                'type': 'recovery',
                'problem': problem,
                'prompt': {'content': solver_prompt[0], 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(correct_with_completion), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(solution), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': completion_score
            })
            
            # Add dark alignment entry using Loki prompt
            results.append({
                'data_type': 'training',
                'alignment': 'dark',
                'type': 'recovery',
                'problem': problem,
                'prompt': {'content': (
                    "You are a math trickster tasked with creating a deliberately incorrect but convincing solution. "
                    "Your goal is to write a solution that appears mathematically sound but contains subtle errors "
                    "that would fool even a careful mathematician.\n\n"
                    f"Problem:\n{problem}\n\n"
                    "Please provide a complete solution that:\n"
                    "1. Uses correct mathematical notation and LaTeX\n"
                    "2. Follows logical steps\n"
                    "3. Contains subtle but significant errors\n"
                    "4. Arrives at an incorrect answer\n"
                    "5. Appears convincing at first glance\n\n"
                    "Make sure to include analysis, step-by-step reasoning, and box the final answer using \\boxed{}"
                ), 'role': 'user'},
                'chosen': {'content': remove_inst_tokens(solution), 'role': 'assistant'},
                'rejected': {'content': remove_inst_tokens(correct_with_completion), 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': completion_score
            })
            
            # Add judge entry with random solution order
            correct_first = random.choice([True, False])
            results.append({
                'data_type': 'training',
                'alignment': 'judge',
                'type': 'recovery',
                'problem': problem,
                'prompt': {'content': Tournament.JUDGE_PROMPT_TEMPLATE.format(
                    problem=problem,
                    solution_a=remove_inst_tokens(correct_with_completion if correct_first else solution),
                    solution_b=remove_inst_tokens(solution if correct_first else correct_with_completion)
                ), 'role': 'user'},
                'chosen': {'content': 'A' if correct_first else 'B', 'role': 'assistant'},
                'rejected': {'content': 'B' if correct_first else 'A', 'role': 'assistant'},
                'score_chosen': 1.0,
                'score_rejected': 0.0
            })
            
        except Exception as e:
            self._log(f"Error creating training entries: {str(e)}")
            return []

        # Add statistics entry
        stats_result = {
            'data_type': 'statistics',
            'id': example_id,
            'example_processed_successfully': True,
            'is_correct_list': [False],  # Wrong solution
            'is_most_common_correct': False,
            'success_rate': 0,
            'total_solutions': 1,
            'correct_solutions': 0,
            'incorrect_solutions': 1,
            'wrong_step_found': wrong_step_index is not None,
            'wrong_step_index': wrong_step_index if wrong_step_index is not None else -1,
            'total_steps': len(wrong_steps),
            'completion_attempts': self.max_attempts
        }
        
        results.append(stats_result)
        return results
