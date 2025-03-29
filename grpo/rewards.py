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
from utils.similarity_checker import SolutionSimilarityChecker
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
        
        # Apply tanh normalization (z-score followed by tanh)
        if len(rewards) > 1:
            # Calculate mean and standard deviation
            self.logger.info(f"Rewards before: {rewards}")
            
            # If mean is negative, clip all rewards from below by zero
            mean_reward = sum(rewards) / len(rewards)
            if mean_reward < 0:
                self.logger.info(f"Mean reward is negative ({mean_reward:.6f}), clipping all rewards to non-negative values")
                rewards = [max(0.0, r) for r in rewards]
                self.logger.info(f"Rewards after clipping: {rewards}")
        
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
        'reward_components': ['base_rewards', 'validation_rewards', 'diversity_bonuses', 'similarity_penalties', 'total_length_penalty'],
        'group_stats': [
            'correct_answers', 'incorrect_answers', 'unique_solutions', 'similar_solutions',
            'total_similarity'
        ]
    }
    
    def __init__(self, config: RewardConfig, similarity_checker: SolutionSimilarityChecker):
        super().__init__(config)
        self.similarity_checker = similarity_checker
        
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
            
            # Extract response parts from completions for similarity calculation
            response_parts = []
            for comp in group_completions:
                response_match = re.search(r'<response>(.*?)</response>', comp, re.DOTALL)
                if response_match:
                    response_parts.append(response_match.group(1))
                else:
                    # If no response tags, use the whole completion
                    response_parts.append(comp)
                    
            # Calculate similarity matrix for group using only response parts
            similarity_matrix = self.similarity_checker.compute_similarity_matrix(response_parts)
            
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
            
            # Calculate similarity only if group has more than one completion
            if len(group_completions) > 1:
                correct_count = sum(all_results)
                incorrect_count = len(group_completions) - correct_count
                    
                # Diversity bonus - NOTE: Always use GPU for similarity calculations (CPU is too slow)
                with torch.no_grad():
                    # Get the similarity row and keep on GPU for performance
                    similarities = similarity_matrix[group_idx].clone()
                    # Set self-similarity to zero
                    similarities[group_idx] = 0
                    
                    # Calculate average on GPU
                    avg_similarity = similarities.mean().item()
                
                self.logger.info(f"Similarity calculation - Average similarity: {avg_similarity:.3f}")
                
                # Calculate diversity bonus/penalty based on difference from threshold
                diff = self.config.group_similarity_threshold - avg_similarity
                diversity_bonus = self.config.group_diversity_bonus * (abs(diff) ** 0.5)
                
                # Initialize diversity_bonus variable for stats tracking
                diversity_bonus_applied = 0.0
                
                if diff > 0:  # Below threshold - more unique
                    if is_correct:
                        reward += diversity_bonus
                        diversity_bonus_applied = diversity_bonus
                        self.stats.reward_components['diversity_bonuses'] = self.stats.reward_components.get('diversity_bonuses', 0) + 1
                        self.logger.info(f"Applied uniqueness bonus: +{diversity_bonus:.3f}")
                else:  # Above threshold - too similar
                    reward -= diversity_bonus
                    diversity_bonus_applied = -diversity_bonus
                    self.stats.reward_components['similarity_penalties'] = self.stats.reward_components.get('similarity_penalties', 0) + 1
                    self.logger.info(f"Applied similarity penalty: -{diversity_bonus:.3f}")
                
                # Update similarity stats with diversity bonus/penalty information
                if diversity_bonus_applied > 0:
                    self.stats.similarity_stats['diversity_bonuses'] = self.stats.similarity_stats.get('diversity_bonuses', 0) + 1
                elif diversity_bonus_applied < 0:
                    self.stats.similarity_stats['similarity_penalties'] = self.stats.similarity_stats.get('similarity_penalties', 0) + 1
                
                # Update group stats with diversity bonus/penalty information
                if diversity_bonus_applied > 0:
                    self.stats.group_stats['diversity_bonuses'] = self.stats.group_stats.get('diversity_bonuses', 0) + 1
                elif diversity_bonus_applied < 0:
                    self.stats.group_stats['similarity_penalties'] = self.stats.group_stats.get('similarity_penalties', 0) + 1
                
            # Update group-specific statistics
            if is_correct:
                self.stats.group_stats['correct_answers'] += 1
            else:
                self.stats.group_stats['incorrect_answers'] += 1
                
            # Initialize variables if they don't exist in the context
            avg_similarity = 0.0
            diversity_bonus_applied = 0.0
            
            # Check if we have similarity information (only available for groups > 1)
            if len(group_completions) > 1:
                # Get similarity information
                similarities = similarity_matrix[group_idx]
                similarities[group_idx] = 0  # Remove self-similarity
                avg_similarity = similarities.mean().item()
                
                # Update similarity stats
                if avg_similarity < self.config.group_similarity_threshold:
                    self.stats.group_stats['unique_solutions'] += 1
                else:
                    self.stats.group_stats['similar_solutions'] += 1
                
                self.stats.group_stats['total_similarity'] += avg_similarity
                
                # Update diversity bonus/penalty stats
                if hasattr(self.stats.group_stats, 'diversity_bonuses') and diversity_bonus_applied > 0:
                    self.stats.group_stats['diversity_bonuses'] += 1
                elif hasattr(self.stats.group_stats, 'similarity_penalties') and diversity_bonus_applied < 0:
                    self.stats.group_stats['similarity_penalties'] += 1
            
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
        'reward_components': ['base_rewards', 'step_continuity_rewards', 'diversity_bonuses', 'similarity_penalties', 'total_length_penalty'],
        'step_stats': ['correct_step_numbering', 'incorrect_step_numbering', 'total_steps_completed'],
        'similarity_stats': ['unique_completions', 'similar_completions', 'total_similarity']
    }
    
    def __init__(self, config: RewardConfig, similarity_checker: SolutionSimilarityChecker = None):
        super().__init__(config)
        self.similarity_checker = similarity_checker

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
            
            # Apply similarity check if we have a similarity checker and group context
            group_completions = kwargs.get('group_completions', [])
            if self.similarity_checker and len(group_completions) > 1:
                # Extract response parts from completions for similarity calculation
                response_parts = []
                for comp in group_completions:
                    response_match = re.search(r'<response>(.*?)</response>', comp, re.DOTALL)
                    if response_match:
                        response_parts.append(response_match.group(1))
                    else:
                        # If no response tags, use the whole completion
                        response_parts.append(comp)
                
                # Calculate similarity matrix
                similarity_matrix = self.similarity_checker.compute_similarity_matrix(response_parts)
                
                # Get the current completion's index in the group
                group_idx = kwargs.get('group_idx', 0)
                
                # Calculate average similarity to other completions
                with torch.no_grad():
                    # Get the similarity row and keep on GPU for performance
                    similarities = similarity_matrix[group_idx].clone()
                    # Set self-similarity to zero
                    similarities[group_idx] = 0
                    
                    # Calculate average on GPU
                    avg_similarity = similarities.mean().item()
                
                self.logger.info(f"Similarity calculation - Average similarity: {avg_similarity:.3f}")
                
                # Calculate diversity bonus/penalty based on difference from threshold
                diff = self.config.group_similarity_threshold - avg_similarity
                diversity_bonus = self.config.group_diversity_bonus * (abs(diff) ** 0.5)
                
                # Initialize diversity_bonus variable for stats tracking
                diversity_bonus_applied = 0.0
                
                if diff > 0:  # Below threshold - more unique
                    if is_correct:
                        reward += diversity_bonus
                        diversity_bonus_applied = diversity_bonus
                        self.stats.reward_components['diversity_bonuses'] = self.stats.reward_components.get('diversity_bonuses', 0) + 1
                        self.logger.info(f"Applied uniqueness bonus: +{diversity_bonus:.3f}")
                else:  # Above threshold - too similar
                    reward -= diversity_bonus
                    diversity_bonus_applied = -diversity_bonus
                    self.stats.reward_components['similarity_penalties'] = self.stats.reward_components.get('similarity_penalties', 0) + 1
                    self.logger.info(f"Applied similarity penalty: -{diversity_bonus:.3f}")
                
                # Update similarity stats
                if avg_similarity < self.config.group_similarity_threshold:
                    self.stats.similarity_stats['unique_completions'] += 1
                else:
                    self.stats.similarity_stats['similar_completions'] += 1
                
                self.stats.similarity_stats['total_similarity'] += avg_similarity
            
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
        """Calculate reward for an architect prompt by testing it with a programming model"""
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
            
            # 3. Test the architect's prompt with a programming model
            try:
                # Create a programming prompt using the architect's guidance
                programming_prompt = f"{PROGRAMMER_SYSTEM_PROMPT_SUB}\n\nProblem:\n{problem}\n\nArchitect's Guidance:\n{architect_response}"
                
                # Get the model using the benchmark config
                programming_model = get_model(self.config,role="main")
                
                # Create a programming agent
                programming_agent = ProgrammingAgent(programming_model)
                
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
                    return reward  # Return early if no code found
                
                # Check code quality
                code_quality_passed, quality_message = check_code_quality(code)
                
                if not code_quality_passed:
                    self.logger.info(f"Code quality check failed: {quality_message}")
                    self.stats.architect_stats['syntax_errors'] += 1
                    return reward  # Return early if code quality check fails
                
                # Run the code and check if it produces a valid output
                execution_success, result, error_message = run_code_safely(code, timeout=self.config.timeout)
                
                if execution_success and result is not None:
                    execution_reward = self.config.execution_reward
                    reward += execution_reward
                    self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                    self.stats.reward_components['execution_valid_architectures'] = self.stats.reward_components.get('execution_valid_architectures', 0) + 1
                    self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
                    
                    # Update architect stats for successful execution
                    self.stats.architect_stats['total_programming_attempts'] += 1
                else:
                    self.logger.info(f"Code execution failed: {error_message}")
                    if "timed out" in error_message:
                        self.stats.architect_stats['timeout_errors'] += 1
                    else:
                        self.stats.architect_stats['execution_errors'] += 1
                    # Update total rewards and average before returning
                    self.stats.reward_components['total_rewards'] += reward
                    total_samples = self.stats.total_batches
                    self.stats.reward_components['average_reward'] = self.stats.reward_components['total_rewards'] / max(1, total_samples)
                    return reward  # Return early if execution fails
                
                # 4. Check if the result matches the correct answer
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
                
                # Compare with tolerance
                is_correct = abs(correct_answer - result) <= self.config.numeric_tolerance
                if is_correct:
                    correctness_reward = self.config.correctness_reward
                    reward += correctness_reward
                    self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                    self.stats.reward_components['correct_architectures'] = self.stats.reward_components.get('correct_architectures', 0) + 1
                    self.stats.architect_stats['correct_architectures'] += 1
                    self.logger.info(f"Applied correctness reward: +{correctness_reward:.3f}")
                    
                    # Update programming success rate
                    total_architectures = self.stats.architect_stats['correct_architectures'] + self.stats.architect_stats['incorrect_architectures']
                    if total_architectures > 0:
                        self.stats.architect_stats['programming_success_rate'] = (
                            self.stats.architect_stats['correct_architectures'] / total_architectures
                        )
                else:
                    self.stats.architect_stats['incorrect_architectures'] += 1
                    self.logger.info(f"Incorrect answer: expected {correct_answer}, got {result}")
                
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


