import re
import asyncio
import torch
import logging
from datetime import datetime
from pathlib import Path
import os, sys
from typing import List, Dict, Tuple, Optional, Any, Union
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.model_utils import *
from utils.agents import *
from utils.solution_utils import (
    extract_numeric_answer, extract_answer_from_solution, 
    extract_code_from_response, check_code_quality, generate_test_cases, run_test_function, run_code_safely)
from abc import ABC, abstractmethod
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from utils.solution_utils import validate_solution
class BaseReward(ABC):
    """Base class for reward calculation
    
    All reward classes handle batches of completions with their corresponding metadata.
    Input format:
    - completions: List[str] - The model outputs to evaluate
    - kwargs: Dict containing lists aligned with completions:
        - prompts: List[str] - The prompts that generated each completion
        - answer: List[str] - The expected answers (may also be 'correct_answer')
        
    The reward calculation process:
    1. Add reward_index to track original completion order
    2. Extract required lists from kwargs (prompts, answers)
    3. Process completions in batches, maintaining order
    4. Return rewards in same order as input completions
    
    Note: The dataset may provide 'answer' or 'correct_answer' - we handle both cases.
          'prompts' is pluralized but other fields maintain dataset names.
    """
    
    __name__ = "base_reward"
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.stats = RewardStats(config)
        self.logger = self._setup_logger()
        
    @abstractmethod
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion
        
        Args:
            completion: The model completion to evaluate
            **kwargs: Additional context needed for reward calculation
            
        Returns:
            float: The calculated reward value
        """
        raise NotImplementedError
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.config.logging_dir) / self.config.model_type
        log_dir.mkdir(exist_ok=True)
        
        logger_name = f'reward_{self.config.model_type}'
        logger = logging.getLogger(logger_name)
        
        # Clear any existing handlers to prevent duplicate logging
        if logger.handlers:
            logger.handlers.clear()
            
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(
            log_dir / f"reward_{timestamp}.log"
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())
        return logger
        
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Synchronous batch processing that runs async code in event loop"""
        # Validate inputs
        prompts = kwargs.get('prompts', [])
        answers = kwargs.get('answer') or kwargs.get('correct_answer', [])
        
        if len(completions) != len(prompts) or len(completions) != len(answers):
            self.logger.error(f"Mismatched lengths: completions={len(completions)}, prompts={len(prompts)}, answers={len(answers)}")
            print("wtf")
            return [0.0] * len(completions)
            
        # Group completions by prompt for group context
        prompt_groups = {}
        for idx, (completion, prompt, ans) in enumerate(zip(completions, prompts, answers)):
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {
                    'completions': [],
                    'answers': [],
                    'indices': []
                }
            prompt_groups[prompt]['completions'].append(completion)
            prompt_groups[prompt]['answers'].append(str(ans))
            prompt_groups[prompt]['indices'].append(idx)
            
        # Process completions in parallel using event loop
        async def process_batch():
            tasks = []
            
            # Extract problems, solutions, and partial solutions from kwargs if present
            problems = kwargs.get('problem', [''] * len(prompts))
            solutions = kwargs.get('model_solution', [''] * len(prompts))
            partial_solutions = kwargs.get('partial_solution', [''] * len(prompts))
            
            # Extract tutor-specific parameters if present
            wrong_steps = kwargs.get('wrong_step', [None] * len(prompts))
            is_corrects = kwargs.get('is_correct', [False] * len(prompts))
            full_solutions = kwargs.get('full_solution', [''] * len(prompts))
            
            for prompt, group in prompt_groups.items():
                # Process each completion in group
                for group_idx, (completion, ans, idx) in enumerate(zip(
                    group['completions'], 
                    group['answers'], 
                    group['indices']
                )):
                    # Create kwargs with group context and original kwargs
                    task_kwargs = {
                        **kwargs,  # Base kwargs first
                        'prompt': prompt,
                        'problem': problems[idx] if idx < len(problems) else '',
                        'solution': solutions[idx] if idx < len(solutions) else '',
                        'partial_solution': partial_solutions[idx] if idx < len(partial_solutions) else '',
                        'answer': str(ans),
                        'group_idx': group_idx,
                        'reward_index': idx,
                        'group_completions': group['completions'],
                        'group_answers': group['answers'], 
                        'group_indices': group['indices'],
                        'wrong_step': wrong_steps[idx] if idx < len(wrong_steps) else None,
                        'is_correct': is_corrects[idx] if idx < len(is_corrects) else False,
                        'full_solution': full_solutions[idx] if idx < len(full_solutions) else ''
                    }
                    task = self.calculate_reward(completion, **task_kwargs)
                    tasks.append(task)
                    
            return await asyncio.gather(*tasks)
            
        # Run async code in event loop - with Jupyter notebook compatibility
        try:
            # Check if we're in IPython/Jupyter
            import sys
            is_jupyter = 'ipykernel' in sys.modules
            
            if is_jupyter:
                self.logger.debug("Detected Jupyter environment, using nest_asyncio")
                # Use nest_asyncio to allow nested event loops in Jupyter
                import nest_asyncio
                nest_asyncio.apply()
                
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                self.logger.debug("Using existing event loop")
            except RuntimeError:
                self.logger.debug("No event loop found - creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the coroutine
            if is_jupyter:
                # In Jupyter, we can directly await the coroutine
                rewards = asyncio.run(process_batch())
            else:
                # In regular Python, use run_until_complete
                rewards = loop.run_until_complete(process_batch())
                
        except Exception as e:
            self.logger.error(f"Error during batch processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            rewards = [0.0] * len(completions)
        
        # Check if any rewards are high enough to be considered good
        if len(rewards) > 1:
            # Calculate max reward
            self.logger.info(f"Rewards before: {rewards}")
            
            # If max reward is below threshold, set all rewards to zero
            max_reward = max(rewards) if rewards else 0
            if max_reward < 1.5:
                self.logger.info(f"Max reward is below threshold ({max_reward:.6f} < 1.5), setting all rewards to zero")
                rewards = [0.0] * len(rewards)
                self.logger.info(f"Rewards after adjustment: {rewards}")
        
        # Update stats and print batch summary
        self.stats.update(rewards, completions=completions, example_type=kwargs.get('example_type', []))
        
        # Print reward-specific statistics summary every batch
        self.logger.info("\nReward Statistics Summary:")
        self.logger.info(self.stats.get_summary(getattr(self, 'relevant_stats', None)))
        
        return rewards
            


class SolutionReward(BaseReward):
    """Reward class for group-based solution evaluation"""
    
    __name__ = "solution_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'validation_rewards', 'total_length_penalty'],
        'group_stats': [
            'correct_answers', 'incorrect_answers'
        ]
    }
    
    def __init__(self, config: RewardConfig, similarity_checker=None):
        super().__init__(config)
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion with group context"""
        try:
            # Get group context from kwargs
            group_completions = kwargs.get('group_completions', [])
            group_answers = kwargs.get('group_answers', [])
            group_indices = kwargs.get('group_indices', [])
            group_idx = kwargs.get('group_idx', 0)
            correct_answer = kwargs.get('answer')
            reward=0.0
            if not all([group_completions, group_answers, group_indices]):
                self.logger.warning(f"Missing required group context - completions: {bool(group_completions)}, answers: {bool(group_answers)}, indices: {bool(group_indices)}")
                return 0.0

            self.logger.info(f"Processing completion {group_idx+1}/{len(group_completions)} in group")
            
            
            # Extract and validate the answer
            model_answer = extract_answer_from_solution(completion)
            if model_answer is None:
                self.logger.info("Thre is not model_answer")
                return reward
                
            # Convert to numeric values
            model_numeric, debug_info = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(str(correct_answer))
            if model_numeric is None or correct_numeric is None:
                self.logger.debug("Could not extract numeric values - returning 0.0")
                return reward
                
                
            # Structure validation rewards
            
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            if not has_thinking or not has_response:
                self.logger.debug("No thinking or response")
                return reward
            validation_reward = 0.0
            # Calculate base reward
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            if is_correct:
                reward += self.config.base_reward
                self.logger.info(f"Applied base reward: +{self.config.base_reward:.3f}")
                self.stats.reward_components['base_rewards'] += 1
                self.stats.reward_components['correct_answers'] += 1
                
            else:
                self.stats.reward_components['incorrect_answers'] += 1


            # Extract response part and validate solution structure
            response_parts = re.findall(r'<response>(.*?)</response>', completion, re.DOTALL)
            if response_parts:
                # Use the validate_solution method to check solution structure
                
                solution_valid, validation_reason = validate_solution(response_parts[0])
                
                if solution_valid:
                    validation_reward += 0.2
                    self.logger.info(f"Solution structure validation passed (+0.2)")
                else:
                    self.logger.info(f"Solution structure validation failed: {validation_reason}")
            
            reward += validation_reward
            if validation_reward > 0:
                self.stats.reward_components['validation_rewards'] += 1
                self.logger.info(f"Applied total validation reward: +{validation_reward:.3f}")
                
            # Extract response section and apply length penalty only to it
            response_parts = re.findall(r'<response>(.*?)</response>', completion, re.DOTALL)
            if response_parts:
                response_length = len(response_parts[0])
                length_penalty = response_length * self.config.length_penalty_factor
                reward -= length_penalty
                self.stats.reward_components['total_length_penalty'] = \
                    self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty

            # Update total rewards and average
            self.stats.reward_components['total_rewards'] += reward
            total_samples = self.stats.reward_components['correct_answers'] + self.stats.reward_components['incorrect_answers']
            self.stats.reward_components['average_reward'] = \
                self.stats.reward_components['total_rewards'] / max(1, total_samples)
            
            
            # Calculate correctness for all completions in group
            all_results = []
            for comp, ans in zip(group_completions, group_answers):
                comp_answer = extract_answer_from_solution(comp)
                if comp_answer is None:
                    all_results.append(False)
                    continue
                    
                comp_numeric, _ = extract_numeric_answer(comp_answer)
                ans_numeric, _ = extract_numeric_answer(ans)
                if comp_numeric is None or ans_numeric is None:
                    all_results.append(False)
                    continue
                    
                all_results.append(abs(comp_numeric - ans_numeric) <= self.config.numeric_tolerance)
            
            # Group information is available but similarity reward is not used
            if len(group_completions) > 1:
                self.logger.info(f"Group information available ({len(group_completions)} completions) but similarity reward disabled")
                
            # Update group-specific statistics
            if is_correct:
                self.stats.group_stats['correct_answers'] += 1
            else:
                self.stats.group_stats['incorrect_answers'] += 1
                
            # Group information is available but not used for similarity rewards
            if len(group_completions) > 1:
                self.logger.info(f"Group has {len(group_completions)} completions, but similarity reward is disabled")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating group reward: {str(e)}")
            return 0.0


class ProgrammingReward(BaseReward):
    """Reward class for programming solution evaluation"""
    
    __name__ = "programming_reward"
    relevant_stats = {
        'reward_components': [
            'syntax_rewards', 'execution_rewards', 'correctness_rewards',
            'total_length_penalty', 'correct_solutions', 'syntax_valid_solutions', 
            'execution_valid_solutions', 'total_rewards', 'average_reward'
        ],
        'programming_stats': [
            'correct_solutions', 'incorrect_solutions', 'syntax_errors', 
            'execution_errors', 'timeout_errors'
        ]
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Initialize programming-specific stats
        self.stats.programming_stats = {
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'syntax_errors': 0,
            'execution_errors': 0,
            'timeout_errors': 0
        }
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a programming solution"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for programming reward calculation")
                return 0.0
            
            # Initialize reward
            reward = 0.0
            
            # 1. Check for thinking and response sections
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if not has_thinking or not has_response:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
                return 0.0
            
            # Extract code from the completion
            # First check if response section exists
            if not has_response:
                self.logger.info("No response section found in completion")
                # We can still try to extract code from the whole completion
                code = extract_code_from_response(completion)
                if not code:
                    self.logger.info("No code found in completion")
                    return reward
            else:
                # Extract code from the response section
                response_match = re.search(r'<response>(.*?)</response>', completion, re.DOTALL)
                response_content = response_match.group(1)
                code = extract_code_from_response(response_content)
                if not code:
                    # If no code in response section, try the whole completion
                    self.logger.info("No code found in response section, trying whole completion")
                    code = extract_code_from_response(completion)
                    if not code:
                        self.logger.info("No code found in completion")
                        return reward
            
            self.logger.info(f"Extracted code length: {len(code)} characters")
            
            # 2. Check code quality (syntax reward/penalty)
            code_quality_passed, quality_message = check_code_quality(code)
            if code_quality_passed:
                syntax_reward = self.config.syntax_reward
                reward += syntax_reward
                self.stats.reward_components['syntax_rewards'] = self.stats.reward_components.get('syntax_rewards', 0) + 1
                self.stats.reward_components['syntax_valid_solutions'] = self.stats.reward_components.get('syntax_valid_solutions', 0) + 1
                self.logger.info(f"Applied syntax reward: +{syntax_reward:.3f}")
            else:
                syntax_penalty = self.config.syntax_penalty
                reward -= syntax_penalty
                self.logger.info(f"Applied syntax penalty: -{syntax_penalty:.3f}")
                self.logger.info(f"Code quality check failed: {quality_message}")
                self.stats.programming_stats['syntax_errors'] += 1
                # Update total rewards and average before returning
                self.stats.reward_components['total_rewards'] += reward
                total_samples = self.stats.total_batches
                self.stats.reward_components['average_reward'] = self.stats.reward_components['total_rewards'] / max(1, total_samples)
                return reward  # Return early if syntax is invalid
            
            # 3. Run the code and check if it produces a valid output (execution reward)
            execution_success, result, error_message = run_code_safely(code, timeout=self.config.timeout)
            if execution_success and result is not None:
                execution_reward = self.config.execution_reward
                reward += execution_reward
                self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                self.stats.reward_components['execution_valid_solutions'] = self.stats.reward_components.get('execution_valid_solutions', 0) + 1
                self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
            else:
                self.logger.info(f"Code execution failed: {error_message}")
                if "timed out" in error_message:
                    self.stats.programming_stats['timeout_errors'] += 1
                else:
                    self.stats.programming_stats['execution_errors'] += 1
                # Update total rewards and average before returning
                self.stats.reward_components['total_rewards'] += reward
                total_samples = self.stats.total_batches
                self.stats.reward_components['average_reward'] = self.stats.reward_components['total_rewards'] / max(1, total_samples)
                return reward  # Return early if execution fails
            
            # 4. Check if the result matches the correct answer (correctness reward)
            # Convert correct_answer to float if it's not already
            try:
                if isinstance(correct_answer, str):
                    numeric_answer, _ = extract_numeric_answer(correct_answer)
                    if numeric_answer is not None:
                        correct_answer = numeric_answer
                    else:
                        correct_answer = float(correct_answer)
                else:
                    correct_answer = float(correct_answer)
            except (ValueError, TypeError):
                self.logger.info(f"Could not convert correct answer to float: {correct_answer}")
                return reward
            
            # Compare with tolerance
            is_correct = abs(correct_answer - result) <= self.config.numeric_tolerance
            if is_correct:
                correctness_reward = self.config.correctness_reward
                reward += correctness_reward
                self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                self.stats.reward_components['correct_solutions'] = self.stats.reward_components.get('correct_solutions', 0) + 1
                self.stats.programming_stats['correct_solutions'] += 1
                self.logger.info(f"Applied correctness reward: +{correctness_reward:.3f}")
            else:
                self.stats.programming_stats['incorrect_solutions'] += 1
                self.logger.info(f"Incorrect answer: expected {correct_answer}, got {result}")
            
            # Apply length penalty
            length_penalty = len(code) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating programming reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0

class FinalizationReward(BaseReward):
    """Reward class for solution finalization evaluation"""
    
    __name__ = "finalization_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'step_continuity_rewards', 'total_length_penalty'],
        'step_stats': ['correct_step_numbering', 'incorrect_step_numbering', 'total_steps_completed']
    }
    
    def __init__(self, config: RewardConfig, similarity_checker=None):
        super().__init__(config)

    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a solution completion"""
        try:
            # Get problem and partial solution
            problem = kwargs.get('problem', '')
            partial_solution = kwargs.get('partial_solution', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, partial_solution, correct_answer]):
                self.logger.warning("Missing required inputs for completion reward calculation")
                return 0.0
            
            # Extract the response part from the completion
            response_match = re.search(r'<response>(.*?)</response>', completion, re.DOTALL)
            if response_match:
                completion_response = response_match.group(1)
            else:
                # If no response tags, give zero reward
                self.logger.info("No response tags found in completion, giving zero reward")
                return 0.0
                
            # Combine partial solution with completion response
            # We don't want the response tag in either part
            full_solution = partial_solution + completion_response
            # Since we're only using the response part in partial solutions,
            # we don't need to check for thinking section
                
            
                
            # Extract and validate the answer
            model_answer = extract_answer_from_solution(full_solution)
            if model_answer is None:
                self.logger.info("No boxed answer found in completion")
                return 0.0
                
            # Convert to numeric values
            model_numeric, _ = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(correct_answer)
            if model_numeric is None or correct_numeric is None:
                self.logger.info(f"Could not extract numeric values - Model: {model_numeric}, Correct: {correct_numeric}")
                return 0.0
                
            # Check correctness
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            self.logger.info(f"Correctness check - Model: {model_numeric:.6f}, Expected: {correct_numeric:.6f}, Correct: {is_correct}")
            
            # Base reward for correct answer
            reward = 0.0
            if is_correct:
                reward = self.config.base_reward
                self.logger.info(f"Applied base reward: +{self.config.base_reward:.3f}")
                self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + 1
                self.stats.reward_components['correct_answers'] = self.stats.reward_components.get('correct_answers', 0) + 1
            else:
                self.stats.reward_components['incorrect_answers'] = self.stats.reward_components.get('incorrect_answers', 0) + 1
                
            
            # Extract steps from completion for logging purposes
            completion_steps = re.findall(r'<step>Step\s+(\d+):', completion, re.IGNORECASE)
            
            # Extract response part from completion
            response_match = re.search(r'<response>(.*?)</response>', completion, re.DOTALL)
            completion_response = response_match.group(1) if response_match else completion
            
            # Get the last step number from the partial solution
            step_numbers = re.findall(r'Step\s*(\d+)[:.)\s]', partial_solution)
            start_step = max([int(num) for num in step_numbers]) if step_numbers else 0
            
            step_continuity_correct, validation_reason = validate_solution(completion_response, start_step=start_step)
            
            if not step_continuity_correct:
                self.logger.info(f"Step validation failed: {validation_reason}")
            
            # Reward for correct step continuity
            if step_continuity_correct:
                continuity_reward = self.config.step_continuity_reward
                reward += continuity_reward
                self.stats.reward_components['step_continuity_rewards'] = self.stats.reward_components.get('step_continuity_rewards', 0) + 1
                self.stats.step_stats['correct_step_numbering'] += 1
                self.logger.info(f"Applied step continuity reward: +{continuity_reward:.3f}")
            else:
                self.stats.step_stats['incorrect_step_numbering'] += 1
            
            # Track total steps completed
            self.stats.step_stats['total_steps_completed'] += len(completion_steps)
            
            # Apply length penalty
            length_penalty = len(completion) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = \
                self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Group information is still passed but similarity reward is not used
            group_completions = kwargs.get('group_completions', [])
            if len(group_completions) > 1:
                # Log that we have group information but not using similarity
                self.logger.info(f"Group information available ({len(group_completions)} completions) but similarity reward disabled")
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.reward_components.get('correct_answers', 0) + self.stats.reward_components.get('incorrect_answers', 0)
            self.stats.reward_components['average_reward'] = \
                self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
                
            return reward
        except Exception as e:
            self.logger.error(f"Error calculating completion reward: {str(e)}")
            return 0.0
        
        
class ArchitectReward(BaseReward):
    """Reward class for architect prompt evaluation"""
    
    __name__ = "architect_reward"
    relevant_stats = {
        'reward_components': [
            'syntax_rewards', 'execution_rewards', 'correctness_rewards',
            'total_length_penalty', 'correct_architectures', 'syntax_valid_architectures', 
            'execution_valid_architectures', 'total_rewards', 'average_reward'
        ],
        'architect_stats': [
            'correct_architectures', 'incorrect_architectures', 'syntax_errors', 
            'execution_errors', 'timeout_errors', 'programming_success_rate',
            'average_programming_score', 'total_programming_attempts'
        ]
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Initialize architect-specific stats
        self.stats.architect_stats = {
            'correct_architectures': 0,
            'incorrect_architectures': 0,
            'syntax_errors': 0,
            'execution_errors': 0,
            'timeout_errors': 0,
            'programming_success_rate': 0.0,
            'average_programming_score': 0.0,
            'total_programming_attempts': 0
        }
        
        # Initialize architect-specific reward components
        self.stats.reward_components.update({
            'syntax_rewards': 0,
            'execution_rewards': 0,
            'correctness_rewards': 0,
            'syntax_valid_architectures': 0,
            'execution_valid_architectures': 0,
            'correct_architectures': 0
        })
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for an architect prompt by testing it with a programming model multiple times"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for architect reward calculation")
                return 0.0
            
            # Initialize reward
            reward = 0.0
            
            # 1. Check for thinking and response sections
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if not has_thinking or not has_response:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
                return 0.0
            
            # Extract response section from the architect's completion
            response_match = re.search(r'<response>(.*?)</response>', completion, re.DOTALL)
            if not response_match:
                self.logger.info("No response section found in architect completion")
                return 0.0
            
            architect_response = response_match.group(1).strip()
            
            # 2. Check response quality (syntax reward)
            # Simple check: response should have some minimum length and structure
            if len(architect_response) > 100 and "Recommended Approach" in architect_response:
                syntax_reward = self.config.syntax_reward
                reward += syntax_reward
                self.stats.reward_components['syntax_rewards'] = self.stats.reward_components.get('syntax_rewards', 0) + 1
                self.stats.reward_components['syntax_valid_architectures'] = self.stats.reward_components.get('syntax_valid_architectures', 0) + 1
                self.logger.info(f"Applied syntax reward: +{syntax_reward:.3f}")
            else:
                syntax_penalty = self.config.syntax_penalty
                reward -= syntax_penalty
                self.logger.info(f"Applied syntax penalty: -{syntax_penalty:.3f}")
                self.logger.info(f"Architect response quality check failed: too short or missing key sections")
                self.stats.architect_stats['syntax_errors'] += 1
                # Update total rewards and average before returning
                self.stats.reward_components['total_rewards'] += reward
                total_samples = self.stats.total_batches
                self.stats.reward_components['average_reward'] = self.stats.reward_components['total_rewards'] / max(1, total_samples)
                return reward  # Return early if syntax is invalid
            
            # 3. Test the architect's prompt with a programming model multiple times
            try:
                # Create a programming prompt using the architect's guidance
                programming_prompt = f"{PROGRAMMER_SYSTEM_PROMPT_SUB}\n\nProblem:\n{problem}\n\nArchitect's Guidance:\n{architect_response}"
                
                # Get the model using the benchmark config
                programming_model = get_model(self.config, role="main")
                
                # Create a programming agent
                programming_agent = ProgrammingAgent(programming_model)
                
                # Make multiple calls to the model and track success rate
                num_attempts = 10  # Number of attempts to make
                successful_attempts = 0
                
                self.logger.info(f"Making {num_attempts} attempts with the programming model")
                
                # Convert correct_answer to float if possible for comparison
                try:
                    if isinstance(correct_answer, str):
                        numeric_answer, _ = extract_numeric_answer(correct_answer)
                        if numeric_answer is not None:
                            correct_answer = numeric_answer
                        else:
                            correct_answer = float(correct_answer)
                    else:
                        correct_answer = float(correct_answer)
                except (ValueError, TypeError):
                    self.logger.info(f"Could not convert correct answer to float: {correct_answer}")
                    return reward
                
                for attempt in range(num_attempts):
                    self.logger.info(f"Attempt {attempt+1}/{num_attempts}")
                    
                    # Generate a programming solution
                    programming_solution = await programming_agent.generate(programming_prompt)
                    
                    # Extract code from the programming solution
                    response_match = re.search(r'<response>(.*?)</response>', programming_solution, re.DOTALL)
                    if response_match:
                        response_content = response_match.group(1)
                        code = extract_code_from_response(response_content)
                        if not code:
                            # If no code in response section, try the whole solution
                            self.logger.info(f"No code found in response section, trying whole solution")
                            code = extract_code_from_response(programming_solution)
                    else:
                        # If no response tags, extract from the whole solution
                        code = extract_code_from_response(programming_solution)
                    
                    self.logger.info(f"Extracted code length: {len(code) if code else 0} characters")
                    
                    if not code:
                        self.logger.info(f"No code found in programming solution")
                        self.stats.architect_stats['execution_errors'] += 1
                        continue  # Skip to next attempt
                    
                    # Check code quality
                    code_quality_passed, quality_message = check_code_quality(code)
                    
                    if not code_quality_passed:
                        self.logger.info(f"Code quality check failed: {quality_message}")
                        self.stats.architect_stats['syntax_errors'] += 1
                        continue  # Skip to next attempt
                    
                    # Run the code and check if it produces a valid output
                    execution_success, result, error_message = run_code_safely(code, timeout=self.config.timeout)
                    
                    if execution_success and result is not None:
                        # Update architect stats for successful execution
                        self.stats.architect_stats['total_programming_attempts'] += 1
                        
                        # Compare with tolerance
                        is_correct = abs(correct_answer - result) <= self.config.numeric_tolerance
                        if is_correct:
                            successful_attempts += 1
                            self.logger.info(f"Attempt {attempt+1} successful: result={result}, expected={correct_answer}")
                        else:
                            self.logger.info(f"Attempt {attempt+1} incorrect: result={result}, expected={correct_answer}")
                    else:
                        self.logger.info(f"Code execution failed: {error_message}")
                        if "timed out" in error_message:
                            self.stats.architect_stats['timeout_errors'] += 1
                        else:
                            self.stats.architect_stats['execution_errors'] += 1
                
                # Calculate success rate
                success_rate = successful_attempts / num_attempts
                self.logger.info(f"Success rate: {successful_attempts}/{num_attempts} = {success_rate:.2f}")
                
                # Apply execution reward based on at least one successful execution
                if successful_attempts > 0:
                    execution_reward = self.config.execution_reward
                    reward += execution_reward
                    self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                    self.stats.reward_components['execution_valid_architectures'] = self.stats.reward_components.get('execution_valid_architectures', 0) + 1
                    self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
                
                # Apply correctness reward based on success rate
                if success_rate > 0:
                    # Scale the correctness reward by the success rate
                    correctness_reward = self.config.correctness_reward * success_rate
                    reward += correctness_reward
                    self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                    self.stats.reward_components['correct_architectures'] = self.stats.reward_components.get('correct_architectures', 0) + 1
                    self.stats.architect_stats['correct_architectures'] += 1
                    self.logger.info(f"Applied correctness reward (scaled by success rate): +{correctness_reward:.3f}")
                    
                    # Update programming success rate
                    total_architectures = self.stats.architect_stats['correct_architectures'] + self.stats.architect_stats['incorrect_architectures']
                    if total_architectures > 0:
                        self.stats.architect_stats['programming_success_rate'] = (
                            self.stats.architect_stats['correct_architectures'] / total_architectures
                        )
                else:
                    self.stats.architect_stats['incorrect_architectures'] += 1
                    self.logger.info(f"No successful attempts")
                
            except Exception as e:
                self.logger.error(f"Error testing architect prompt: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.stats.architect_stats['execution_errors'] += 1
            
            # Apply length penalty
            length_penalty = len(architect_response) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components['total_rewards'] / max(1, total_samples)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating architect reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0


class TestProgrammingReward(BaseReward):
    """Reward class for test function evaluation"""
    
    __name__ = "test_programming_reward"
    relevant_stats = {
        'reward_components': [
            'syntax_rewards', 'execution_rewards', 'correctness_rewards',
            'total_length_penalty', 'correct_tests', 'syntax_valid_tests', 
            'execution_valid_tests', 'total_rewards', 'average_reward'
        ],
        'test_programming_stats': [
            'correct_tests', 'incorrect_tests', 'syntax_errors', 
            'execution_errors', 'timeout_errors'
        ]
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Initialize test-programming-specific stats
        self.stats.test_programming_stats = {
            'correct_tests': 0,
            'incorrect_tests': 0,
            'syntax_errors': 0,
            'execution_errors': 0,
            'timeout_errors': 0
        }
        
        # Initialize test-programming-specific reward components
        self.stats.reward_components.update({
            'syntax_rewards': 0,
            'execution_rewards': 0,
            'correctness_rewards': 0,
            'syntax_valid_tests': 0,
            'execution_valid_tests': 0,
            'correct_tests': 0
        })
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a test function solution"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for test programming reward calculation")
                return 0.0
            
            # Initialize reward
            reward = 0.0
            
            # 1. Check for thinking and response sections
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if not has_thinking or not has_response:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
                return 0.0
            
            # Extract test function from the completion
            test_function = extract_code_from_response(completion)
            if not test_function:
                self.logger.info("No test function found in completion")
                return reward
            
            self.logger.info(f"Extracted test function length: {len(test_function)} characters")
            
            # 2. Check code quality (syntax reward/penalty)
            code_quality_passed, quality_message = check_code_quality(test_function)
            if code_quality_passed:
                syntax_reward = self.config.syntax_reward
                reward += syntax_reward
                self.stats.reward_components['syntax_rewards'] = self.stats.reward_components.get('syntax_rewards', 0) + 1
                self.stats.reward_components['syntax_valid_tests'] = self.stats.reward_components.get('syntax_valid_tests', 0) + 1
                self.logger.info(f"Applied syntax reward: +{syntax_reward:.3f}")
            else:
                syntax_penalty = self.config.syntax_penalty
                reward -= syntax_penalty
                self.logger.info(f"Applied syntax penalty: -{syntax_penalty:.3f}")
                self.logger.info(f"Code quality check failed: {quality_message}")
                self.stats.test_programming_stats['syntax_errors'] += 1
                # Update total rewards and average before returning
                self.stats.reward_components['total_rewards'] += reward
                total_samples = self.stats.total_batches
                self.stats.reward_components['average_reward'] = self.stats.reward_components['total_rewards'] / max(1, total_samples)
                return reward  # Return early if syntax is invalid
            
            # 3. Generate test cases
            try:
                # Convert correct_answer to float if it's not already
                if isinstance(correct_answer, str):
                    numeric_answer, _ = extract_numeric_answer(correct_answer)
                    if numeric_answer is not None:
                        correct_answer = numeric_answer
                    else:
                        correct_answer = float(correct_answer)
                else:
                    correct_answer = float(correct_answer)
                    
                # Generate test cases including the correct answer and many incorrect answers (50 by default)
                test_cases = generate_test_cases(correct_answer, num_cases=50)
                self.logger.info(f"Generated {len(test_cases)} test cases for validation")
            except (ValueError, TypeError):
                self.logger.info(f"Could not convert correct answer to float: {correct_answer}")
                return reward
            
            # 4. Run the test function on all test cases
            success, results, error_message = run_test_function(
                test_function, 
                test_cases, 
                correct_answer,
                timeout=self.config.timeout
            )
            
            if success:
                execution_reward = self.config.execution_reward
                reward += execution_reward
                self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                self.stats.reward_components['execution_valid_tests'] = self.stats.reward_components.get('execution_valid_tests', 0) + 1
                self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
                
                # Add correctness reward for a valid test function
                correctness_reward = self.config.correctness_reward
                reward += correctness_reward
                self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                self.stats.reward_components['correct_tests'] = self.stats.reward_components.get('correct_tests', 0) + 1
                self.stats.test_programming_stats['correct_tests'] += 1
                self.logger.info(f"Applied correctness reward: +{correctness_reward:.3f}")
            else:
                self.logger.info(f"Test function validation failed: {error_message}")
                self.stats.test_programming_stats['incorrect_tests'] += 1
                if "timed out" in error_message:
                    self.stats.test_programming_stats['timeout_errors'] += 1
                else:
                    self.stats.test_programming_stats['execution_errors'] += 1
            
            # Apply length penalty
            length_penalty = len(test_function) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Calculate and update test function success metrics
            if success:
                # Count test cases that passed/failed
                passed_cases = sum(1 for result in results.values() if result is True)
                total_cases = len(results)
                
                # Update test case statistics
                self.stats.test_programming_stats['test_cases_passed'] += passed_cases
                self.stats.test_programming_stats['total_test_cases_evaluated'] += total_cases
                
                # Calculate and update success rate
                total_tests = self.stats.test_programming_stats['correct_tests'] + self.stats.test_programming_stats['incorrect_tests']
                if total_tests > 0:
                    self.stats.test_programming_stats['test_function_success_rate'] = (
                        self.stats.test_programming_stats['correct_tests'] / total_tests
                    )
                
                # Calculate and update average test cases passed
                if self.stats.test_programming_stats['total_test_cases_evaluated'] > 0:
                    self.stats.test_programming_stats['average_test_cases_passed'] = (
                        self.stats.test_programming_stats['test_cases_passed'] / 
                        self.stats.test_programming_stats['total_test_cases_evaluated']
                    )
                
                # Log detailed test case results
                self.logger.info(f"Test function passed {passed_cases}/{total_cases} test cases")
                self.logger.info(f"Current test function success rate: {self.stats.test_programming_stats['test_function_success_rate']:.2%}")
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating test programming reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0


class TutorReward(BaseReward):
    """Reward class for tutor evaluation of solutions and identification of errors"""
    
    __name__ = "tutor_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'correct_verdict_rewards', 'correct_fix_rewards', 'total_length_penalty'],
        'tutor_stats': ['correct_verdicts', 'incorrect_verdicts', 'correct_fixes', 'incorrect_fixes']
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Initialize tutor-specific stats
        self.stats.tutor_stats = {
            'correct_verdicts': 0,
            'incorrect_verdicts': 0,
            'correct_fixes': 0,
            'incorrect_fixes': 0
        }
        
        # Initialize tutor-specific reward components
        self.stats.reward_components.update({
            'correct_verdict_rewards': 0,
            'correct_fix_rewards': 0
        })
        
    def _extract_evaluation(self, completion: str) -> bool:
        """Extract whether the tutor evaluated the solution as correct or incorrect"""
        # First check for verdict tags
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', completion, re.DOTALL)
        if verdict_match:
            verdict_content = verdict_match.group(1).strip().lower()
            # Check for positive indicators
            if any(word in verdict_content for word in ['correct', 'right', 'valid', 'solution is good']):
                return True
            # Check for negative indicators
            elif any(word in verdict_content for word in ['incorrect', 'wrong', 'error', 'mistake']):
                return False
            # Check for "step X" pattern which indicates an error
            elif re.search(r'step\s+\d+', verdict_content):
                return False
        
        # Default to None if we can't determine
        return None
        
    def _extract_wrong_step(self, completion: str) -> Optional[int]:
        """
        Extract which step the tutor identified as wrong.
        If the verdict contains exactly one integer, return that integer.
        Otherwise return None.
        """
        # First check for verdict tags
        verdict_match = re.search(r'<verdict>(.*?)</verdict>', completion, re.DOTALL)
        if verdict_match:
            verdict_content = verdict_match.group(1).strip()
            
            # Find all integers in the verdict
            integers = re.findall(r'\b\d+\b', verdict_content)
            
            # If there's exactly one integer, return it
            if len(integers) == 1:
                try:
                    return int(integers[0])
                except (ValueError, IndexError):
                    pass
        return None

    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a tutor evaluation"""
        try:
            # Get problem, full solution, and wrong step information
            problem = kwargs.get('problem', '')
            full_solution = kwargs.get('full_solution', '')
            is_correct = kwargs.get('is_correct')
            
            # Try to convert wrong_step to int, default to None if not possible
            wrong_step = kwargs.get('wrong_step', None)
            if wrong_step is not None:
                try:
                    wrong_step = int(wrong_step)
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert wrong_step to int: {wrong_step}")
                    wrong_step = None
            
            if not all([problem, full_solution]):
                self.logger.warning("Missing required inputs for tutor reward calculation")
                return 0.0
                
            # If is_correct is None, we need to infer it from the model's evaluation
            if is_correct is None:
                self.logger.info("No ground truth 'is_correct' value provided, will infer from model evaluation")
            
            # Initialize reward
            reward = 0.0
            
            # Check for thinking and response tags
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if not has_thinking or not has_response:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
                return 0.0
            
            # Extract response section
            response_match = re.search(r'<response>(.*?)</response>', completion, re.DOTALL)
            if not response_match:
                self.logger.info("Could not extract response section")
                return 0.0
            
            response_content = response_match.group(1)
            
            # Check for verdict tags in the response
            verdict_match = re.search(r'<verdict>(.*?)</verdict>', response_content, re.DOTALL)
            if not verdict_match:
                self.logger.info("No verdict tags found in response")
                return 0.0
            
            verdict_content = verdict_match.group(1).strip()
            
            # Check if the verdict is correct
            verdict_is_correct = False
            
            # Extract the model's evaluation of the solution using the helper methods
            model_says_correct = self._extract_evaluation(completion)
            identified_step = self._extract_wrong_step(completion)
            
            if is_correct:
                # Solution is correct, verdict should indicate correctness
                if model_says_correct:
                    verdict_is_correct = True
                    self.stats.tutor_stats['correct_verdicts'] += 1
                    self.logger.info("Correct verdict for correct solution")
                else:
                    self.stats.tutor_stats['incorrect_verdicts'] += 1
                    self.logger.info(f"Incorrect verdict for correct solution: {verdict_content}")
            else:
                # Solution has error, verdict should be "Step X"
                if identified_step is not None:
                    if wrong_step is not None and identified_step == wrong_step:
                        verdict_is_correct = True
                        self.stats.tutor_stats['correct_verdicts'] += 1
                        self.logger.info(f"Correct verdict identifying wrong step {wrong_step}")
                    else:
                        self.stats.tutor_stats['incorrect_verdicts'] += 1
                        self.logger.info(f"Incorrect verdict: identified step {identified_step}, actual wrong step {wrong_step}")
                else:
                    self.stats.tutor_stats['incorrect_verdicts'] += 1
                    self.logger.info(f"Incorrect verdict format for solution with error: {verdict_content}")
            
            # Award points for correct verdict
            if verdict_is_correct is True:  # Explicitly check for True since it could be None
                verdict_reward = self.config.tutor_verdict_reward
                reward += verdict_reward
                self.stats.reward_components['correct_verdict_rewards'] = self.stats.reward_components.get('correct_verdict_rewards', 0) + 1
                self.logger.info(f"Applied verdict reward: +{verdict_reward:.3f}")
            
            # If the solution has an error or we don't know, check the finalization
            if is_correct is False or (is_correct is None and not model_says_correct):
                # Extract finalization section
                finalization_match = re.search(r'<finalization>(.*?)</finalization>', response_content, re.DOTALL)
                if finalization_match:
                    finalization_content = finalization_match.group(1).strip()
                    
                    # If finalization is not empty, check if it produces the correct answer
                    if finalization_content:
                        # Extract the answer from the finalization
                        model_answer = extract_answer_from_solution(finalization_content)
                        if model_answer is not None:
                            # Get the correct answer from the original problem
                            correct_answer = kwargs.get('answer', '')
                            
                            # Convert to numeric values
                            model_numeric, _ = extract_numeric_answer(model_answer)
                            correct_numeric, _ = extract_numeric_answer(correct_answer)
                            
                            if model_numeric is not None and correct_numeric is not None:
                                # Check if the fixed solution is correct
                                fix_is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
                                
                                if fix_is_correct:
                                    # Award points for correct fix
                                    fix_reward = self.config.tutor_fix_reward
                                    
                                    # If both the verdict and fix are correct, award bonus points
                                    if verdict_is_correct:
                                        fix_reward = self.config.tutor_combined_reward
                                    
                                    reward += fix_reward
                                    self.stats.reward_components['correct_fix_rewards'] = self.stats.reward_components.get('correct_fix_rewards', 0) + 1
                                    self.stats.tutor_stats['correct_fixes'] += 1
                                    self.logger.info(f"Applied fix reward: +{fix_reward:.3f}")
                                else:
                                    self.stats.tutor_stats['incorrect_fixes'] += 1
                                    self.logger.info(f"Incorrect fix: expected {correct_numeric}, got {model_numeric}")
                            else:
                                self.logger.info("Could not extract numeric values from fix or correct answer")
                        else:
                            self.logger.info("No boxed answer found in finalization")
                    else:
                        self.logger.info("Empty finalization section")
                else:
                    self.logger.info("No finalization tags found in response")
            
            # Apply length penalty
            length_penalty = len(completion) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = \
                self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            # Log detailed reward breakdown
            self.logger.info(f"Tutor reward breakdown: verdict_correct={verdict_is_correct} ({self.config.tutor_verdict_reward if verdict_is_correct else 0}), " +
                            f"fix_reward={fix_reward if 'fix_reward' in locals() else 0}, " +
                            f"length_penalty={length_penalty:.4f}, total={reward:.4f}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating tutor reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0


class DualProofReward(BaseReward):
    """Reward class for dual proof evaluation (logical proof + programming solution)"""
    
    __name__ = "dual_proof_reward"
    relevant_stats = {
        'reward_components': [
            'proof_rewards', 'code_rewards', 'structure_rewards',
            'total_length_penalty', 'correct_proofs', 'correct_code', 
            'correct_dual_solutions', 'total_rewards', 'average_reward'
        ],
        'dual_proof_stats': [
            'correct_proofs', 'incorrect_proofs', 'correct_code', 
            'incorrect_code', 'correct_dual_solutions', 'structure_errors',
            'syntax_errors', 'execution_errors', 'timeout_errors'
        ]
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Initialize test-programming-specific stats
        self.stats.test_programming_stats = {
            'correct_tests': 0,
            'incorrect_tests': 0,
            'syntax_errors': 0,
            'execution_errors': 0,
            'timeout_errors': 0
        }
        
        # Initialize test-programming-specific reward components
        self.stats.reward_components.update({
            'syntax_rewards': 0,
            'execution_rewards': 0,
            'correctness_rewards': 0,
            'syntax_valid_tests': 0,
            'execution_valid_tests': 0,
            'correct_tests': 0
        })
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a test function solution"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for test programming reward calculation")
                return 0.0
            
            # Initialize reward
            reward = 0.0
            
            # 1. Check for thinking and response sections
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if not has_thinking or not has_response:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
                return 0.0
            
            # Extract test function from the completion
            test_function = extract_code_from_response(completion)
            if not test_function:
                self.logger.info("No test function found in completion")
                return reward
            
            self.logger.info(f"Extracted test function length: {len(test_function)} characters")
            
            # 2. Check code quality (syntax reward/penalty)
            code_quality_passed, quality_message = check_code_quality(test_function)
            if code_quality_passed:
                syntax_reward = self.config.syntax_reward
                reward += syntax_reward
                self.stats.reward_components['syntax_rewards'] = self.stats.reward_components.get('syntax_rewards', 0) + 1
                self.stats.reward_components['syntax_valid_tests'] = self.stats.reward_components.get('syntax_valid_tests', 0) + 1
                self.logger.info(f"Applied syntax reward: +{syntax_reward:.3f}")
            else:
                syntax_penalty = self.config.syntax_penalty
                reward -= syntax_penalty
                self.logger.info(f"Applied syntax penalty: -{syntax_penalty:.3f}")
                self.logger.info(f"Code quality check failed: {quality_message}")
                self.stats.test_programming_stats['syntax_errors'] += 1
                # Update total rewards and average before returning
                self.stats.reward_components['total_rewards'] += reward
                total_samples = self.stats.total_batches
                self.stats.reward_components['average_reward'] = self.stats.reward_components['total_rewards'] / max(1, total_samples)
                return reward  # Return early if syntax is invalid
            
            # 3. Generate test cases
            try:
                # Convert correct_answer to float if it's not already
                if isinstance(correct_answer, str):
                    numeric_answer, _ = extract_numeric_answer(correct_answer)
                    if numeric_answer is not None:
                        correct_answer = numeric_answer
                    else:
                        correct_answer = float(correct_answer)
                else:
                    correct_answer = float(correct_answer)
                    
                # Generate test cases including the correct answer and many incorrect answers (50 by default)
                test_cases = generate_test_cases(correct_answer, num_cases=50)
                self.logger.info(f"Generated {len(test_cases)} test cases for validation")
            except (ValueError, TypeError):
                self.logger.info(f"Could not convert correct answer to float: {correct_answer}")
                return reward
            
            # 4. Run the test function on all test cases
            success, results, error_message = run_test_function(
                test_function, 
                test_cases, 
                correct_answer,
                timeout=self.config.timeout
            )
            
            if success:
                execution_reward = self.config.execution_reward
                reward += execution_reward
                self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                self.stats.reward_components['execution_valid_tests'] = self.stats.reward_components.get('execution_valid_tests', 0) + 1
                self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
                
                # Add correctness reward for a valid test function
                correctness_reward = self.config.correctness_reward
                reward += correctness_reward
                self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                self.stats.reward_components['correct_tests'] = self.stats.reward_components.get('correct_tests', 0) + 1
                self.stats.test_programming_stats['correct_tests'] += 1
                self.logger.info(f"Applied correctness reward: +{correctness_reward:.3f}")
            else:
                self.logger.info(f"Test function validation failed: {error_message}")
                self.stats.test_programming_stats['incorrect_tests'] += 1
                if "timed out" in error_message:
                    self.stats.test_programming_stats['timeout_errors'] += 1
                else:
                    self.stats.test_programming_stats['execution_errors'] += 1
            
            # Apply length penalty
            length_penalty = len(test_function) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Calculate and update test function success metrics
            if success:
                # Count test cases that passed/failed
                passed_cases = sum(1 for result in results.values() if result is True)
                total_cases = len(results)
                
                # Update test case statistics
                self.stats.test_programming_stats['test_cases_passed'] += passed_cases
                self.stats.test_programming_stats['total_test_cases_evaluated'] += total_cases
                
                # Calculate and update success rate
                total_tests = self.stats.test_programming_stats['correct_tests'] + self.stats.test_programming_stats['incorrect_tests']
                if total_tests > 0:
                    self.stats.test_programming_stats['test_function_success_rate'] = (
                        self.stats.test_programming_stats['correct_tests'] / total_tests
                    )
                
                # Calculate and update average test cases passed
                if self.stats.test_programming_stats['total_test_cases_evaluated'] > 0:
                    self.stats.test_programming_stats['average_test_cases_passed'] = (
                        self.stats.test_programming_stats['test_cases_passed'] / 
                        self.stats.test_programming_stats['total_test_cases_evaluated']
                    )
                
                # Log detailed test case results
                self.logger.info(f"Test function passed {passed_cases}/{total_cases} test cases")
                self.logger.info(f"Current test function success rate: {self.stats.test_programming_stats['test_function_success_rate']:.2%}")
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating test programming reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0


class TestDrivenProgrammerReward(BaseReward):
    """Reward class for test-driven programmer evaluation (test suite + implementation)"""
    
    __name__ = "test_driven_programmer_reward"
    relevant_stats = {
        'reward_components': [
            'test_rewards', 'implementation_rewards', 'structure_rewards',
            'total_length_penalty', 'correct_tests', 'correct_implementations', 
            'correct_test_driven_solutions', 'total_rewards', 'average_reward'
        ],
        'test_driven_programmer_stats': [
            'correct_tests', 'incorrect_tests', 'correct_implementations', 
            'incorrect_implementations', 'correct_test_driven_solutions', 'structure_errors',
            'syntax_errors', 'execution_errors', 'timeout_errors', 'test_coverage'
        ]
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Initialize test-driven-programmer-specific stats
        self.stats.test_driven_programmer_stats = {
            'correct_tests': 0,
            'incorrect_tests': 0,
            'correct_implementations': 0,
            'incorrect_implementations': 0,
            'correct_test_driven_solutions': 0,
            'structure_errors': 0,
            'syntax_errors': 0,
            'execution_errors': 0,
            'timeout_errors': 0,
            'test_coverage': 0.0
        }
        
        # Initialize test-driven-programmer-specific reward components
        self.stats.reward_components.update({
            'test_rewards': 0,
            'implementation_rewards': 0,
            'structure_rewards': 0,
            'correct_tests': 0,
            'correct_implementations': 0,
            'correct_test_driven_solutions': 0
        })
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a test-driven programmer solution (test suite + implementation)"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for test-driven programmer reward calculation")
                return 0.0
            
            # Initialize reward
            reward = 0.0
            
            # 1. Check for thinking and response sections
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if not has_thinking or not has_response:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
                return 0.0
            
            # 2. Check for test and implementation sections within response
            response_match = re.search(r'<response>(.*?)</response>', completion, re.DOTALL)
            if not response_match:
                self.logger.info("No response section found in completion")
                return 0.0
                
            response_content = response_match.group(1)
            
            has_test = bool(re.search(r'<test>.*?</test>', response_content, re.DOTALL))
            has_implementation = bool(re.search(r'<implementation>.*?</implementation>', response_content, re.DOTALL))
            
            if not has_test or not has_implementation:
                self.logger.info(f"Missing {'test' if not has_test else ''} {'implementation' if not has_implementation else ''} section(s) in response")
                # Apply structure penalty
                self.stats.test_driven_programmer_stats['structure_errors'] += 1
                return 0.0
            
            # Apply structure reward for having all required sections
            structure_reward = 0.2
            reward += structure_reward
            self.stats.reward_components['structure_rewards'] = self.stats.reward_components.get('structure_rewards', 0) + 1
            self.logger.info(f"Applied structure reward: +{structure_reward:.3f}")
            
            # 3. Extract and evaluate the test suite
            test_match = re.search(r'<test>(.*?)</test>', response_content, re.DOTALL)
            test_content = test_match.group(1) if test_match else ""
            
            if not test_content:
                self.logger.info("No test content found in test section")
                return reward  # Return with just the structure reward
            
            # Check test quality (syntax)
            test_quality_passed, test_quality_message = check_code_quality(test_content)
            test_correct = False
            
            if test_quality_passed:
                self.logger.info("Test syntax check passed")
                # Award partial reward for syntactically correct tests
                test_syntax_reward = self.config.base_reward / 8
                reward += test_syntax_reward
                self.logger.info(f"Applied test syntax reward: +{test_syntax_reward:.3f}")
            else:
                self.logger.info(f"Test syntax check failed: {test_quality_message}")
                self.stats.test_driven_programmer_stats['syntax_errors'] += 1
                # Continue to implementation check even if tests have syntax errors
            
            # 4. Extract and evaluate the implementation
            implementation_match = re.search(r'<implementation>(.*?)</implementation>', response_content, re.DOTALL)
            implementation = implementation_match.group(1) if implementation_match else ""
            
            if not implementation:
                self.logger.info("No implementation found in implementation section")
                return reward  # Return with just the test reward if applicable
            
            # Check implementation quality (syntax)
            implementation_quality_passed, implementation_quality_message = check_code_quality(implementation)
            implementation_correct = False
            
            if not implementation_quality_passed:
                self.logger.info(f"Implementation quality check failed: {implementation_quality_message}")
                self.stats.test_driven_programmer_stats['syntax_errors'] += 1
                # Continue with test evaluation even if implementation has syntax errors
            else:
                # Run the implementation and check if it produces a valid output
                execution_success, result, error_message = run_code_safely(implementation, timeout=self.config.timeout)
                
                if execution_success and result is not None:
                    # Check if the implementation result matches the correct answer
                    try:
                        if isinstance(correct_answer, str):
                            numeric_answer, _ = extract_numeric_answer(correct_answer)
                            if numeric_answer is not None:
                                correct_answer = numeric_answer
                            else:
                                correct_answer = float(correct_answer)
                        else:
                            correct_answer = float(correct_answer)
                            
                        # Compare with tolerance
                        implementation_correct = abs(correct_answer - result) <= self.config.numeric_tolerance
                        if implementation_correct:
                            # Award 1/4 of base reward for correct implementation
                            implementation_reward = self.config.base_reward / 4
                            reward += implementation_reward
                            self.stats.reward_components['implementation_rewards'] = self.stats.reward_components.get('implementation_rewards', 0) + 1
                            self.stats.reward_components['correct_implementations'] = self.stats.reward_components.get('correct_implementations', 0) + 1
                            self.stats.test_driven_programmer_stats['correct_implementations'] += 1
                            self.logger.info(f"Applied implementation reward: +{implementation_reward:.3f}")
                        else:
                            self.stats.test_driven_programmer_stats['incorrect_implementations'] += 1
                            self.logger.info(f"Incorrect implementation result: expected {correct_answer}, got {result}")
                    except (ValueError, TypeError):
                        self.logger.info(f"Could not convert correct answer to float: {correct_answer}")
                else:
                    self.logger.info(f"Implementation execution failed: {error_message}")
                    if "timed out" in error_message:
                        self.stats.test_driven_programmer_stats['timeout_errors'] += 1
                    else:
                        self.stats.test_driven_programmer_stats['execution_errors'] += 1
            
            # 5. Evaluate the test suite independently
            # First, try to run the test suite on its own to check syntax and basic functionality
            test_only_code = f"""
{test_content}

# Dummy implementation that returns the correct answer
def solution():
    return {correct_answer}

# Run tests if this file is executed directly
if __name__ == '__main__':
    import unittest
    unittest.main()
"""
            
            test_syntax_success, _, test_syntax_error = run_code_safely(test_only_code, timeout=self.config.timeout)
            
            if test_syntax_success:
                self.logger.info("Test suite runs successfully with a dummy implementation")
                
                # Now try to run the tests with the actual implementation if it's syntactically valid
                if implementation_quality_passed:
                    # Combine test and implementation into a single file for execution
                    combined_code = f"""
{test_content}

# Implementation
{implementation}

# Run tests if this file is executed directly
if __name__ == '__main__':
    import unittest
    unittest.main()
"""
                    
                    # Run the combined code to see if tests pass
                    test_execution_success, _, test_error_message = run_code_safely(combined_code, timeout=self.config.timeout)
                    
                    if test_execution_success:
                        # Tests ran without errors with the actual implementation
                        test_correct = True
                    else:
                        self.logger.info(f"Tests failed with the actual implementation: {test_error_message}")
                        # This is expected if the implementation is incorrect but tests are good
                        if not implementation_correct:
                            # If implementation is wrong but tests caught it, that's good!
                            test_correct = True
                            self.logger.info("Tests correctly identified an incorrect implementation")
                        else:
                            # If implementation is correct but tests fail, tests are wrong
                            test_correct = False
                            self.logger.info("Tests incorrectly fail on a correct implementation")
                else:
                    # If implementation has syntax errors, we can't run the tests with it
                    # But the tests themselves are syntactically valid, which is good
                    test_correct = True
                    self.logger.info("Tests are syntactically valid, but can't be run with invalid implementation")
                
                if test_correct:
                    # Award 1/4 of base reward for correct tests
                    test_reward = self.config.base_reward / 4
                    reward += test_reward
                    self.stats.reward_components['test_rewards'] = self.stats.reward_components.get('test_rewards', 0) + 1
                    self.stats.reward_components['correct_tests'] = self.stats.reward_components.get('correct_tests', 0) + 1
                    self.stats.test_driven_programmer_stats['correct_tests'] += 1
                    self.logger.info(f"Applied test reward: +{test_reward:.3f}")
                else:
                    self.stats.test_driven_programmer_stats['incorrect_tests'] += 1
            else:
                self.logger.info(f"Test suite has runtime errors: {test_syntax_error}")
                self.stats.test_driven_programmer_stats['incorrect_tests'] += 1
            
            # 6. Award bonus if both components are correct
            if implementation_correct and test_correct:
                bonus_reward = self.config.base_reward / 2  # Additional 1/2 of base reward for having both correct
                reward += bonus_reward
                self.stats.reward_components['correct_test_driven_solutions'] = self.stats.reward_components.get('correct_test_driven_solutions', 0) + 1
                self.stats.test_driven_programmer_stats['correct_test_driven_solutions'] += 1
                self.logger.info(f"Applied test-driven solution bonus: +{bonus_reward:.3f}")
            
            # Apply length penalty
            length_penalty = len(completion) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating test-driven programmer reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0


