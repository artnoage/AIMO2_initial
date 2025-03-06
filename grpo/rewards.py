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
from utils.solution_utils import (
    extract_numeric_answer, extract_answer_from_solution, 
    extract_code_from_response, check_code_quality, run_code_safely,
    count_manual_steps)
from utils.similarity_checker import SolutionSimilarityChecker
from abc import ABC, abstractmethod
from config import RewardConfig
from reward_stats import RewardStats

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
                        'problem': problems[idx],  # Map to original index
                        'solution': solutions[idx], # Map to original index
                        'partial_solution': partial_solutions[idx], # Map to original index
                        'answer': str(ans),
                        'group_idx': group_idx,
                        'reward_index': idx,
                        'group_completions': group['completions'],
                        'group_answers': group['answers'], 
                        'group_indices': group['indices']
                    }
                    task = self.calculate_reward(completion, **task_kwargs)
                    tasks.append(task)
                    
            return await asyncio.gather(*tasks)
            
        # Run async code in event loop
        # Get or create event loop for async processing
        try:
            loop = asyncio.get_event_loop()
            self.logger.debug("Using existing event loop")
        except RuntimeError:
            self.logger.debug("No event loop found - creating new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            rewards = loop.run_until_complete(process_batch())
        except Exception as e:
            self.logger.error(f"Error during batch processing: {str(e)}")
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
    
    __name__ = "group_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'diversity_bonuses', 'similarity_penalties'],
        'group_stats': [
            'correct_answers', 'incorrect_answers', 'unique_solutions', 'similar_solutions',
            'total_similarity'
        ]
    }
    
    def __init__(self, config: RewardConfig, similarity_checker: SolutionSimilarityChecker):
        super().__init__(config)
        self.similarity_checker = similarity_checker
        # Initialize the wait logger for tracking "wait a second" moments
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion with group context"""
        try:
            # Get group context from kwargs
            group_completions = kwargs.get('group_completions', [])
            group_answers = kwargs.get('group_answers', [])
            group_indices = kwargs.get('group_indices', [])
            group_idx = kwargs.get('group_idx', 0)
            correct_answer = kwargs.get('answer')
            prompt = kwargs.get('prompt', '')
            problem = kwargs.get('problem', '')
            example_type = kwargs.get('example_type', '')
            
            if not all([group_completions, group_answers, group_indices]):
                self.logger.warning(f"Missing required group context - completions: {bool(group_completions)}, answers: {bool(group_answers)}, indices: {bool(group_indices)}")
                return 0.0

            self.logger.info(f"Processing completion {group_idx+1}/{len(group_completions)} in group")
            
            
            # Extract and validate the answer
            model_answer = extract_answer_from_solution(completion)
            if model_answer is None:
                return 0.0
                
            # Convert to numeric values
            model_numeric, debug_info = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(str(correct_answer))
            if model_numeric is None or correct_numeric is None:
                self.logger.debug("Could not extract numeric values - returning 0.0")
                return 0.0
                
            # Calculate base reward
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            reward = self.config.group_base_reward if is_correct else 0.0
            if is_correct:
                reward = self.config.base_reward
                self.logger.info(f"Applied base reward: +{self.config.base_reward:.3f}")
                self.stats.reward_components['base_rewards'] += 1
                self.stats.reward_components['correct_answers'] += 1
                
            else:
                self.stats.reward_components['incorrect_answers'] += 1
                
            # Structure validation rewards
            validation_reward = 0.0
            
            # Count valid steps
            step_count = count_manual_steps(completion)
            if step_count > 0:
                
                # Check step ordering and uniqueness
                response_parts = re.findall(r'<response>(.*?)</response>', completion, re.DOTALL)
                if response_parts:
                    # Count occurrences of each step number
                    step_counts = {i: len(re.findall(rf'Step\s*{i}[:.)\s]', response_parts[0], re.IGNORECASE)) 
                                 for i in range(1, step_count + 1)}
                    
                    # Check if steps are properly closed
                    opening_tags = len(re.findall(r'<step>', response_parts[0], re.IGNORECASE))
                    closing_tags = len(re.findall(r'</step>', response_parts[0], re.IGNORECASE))
                    steps_properly_closed = opening_tags == closing_tags
                    
                    # Extract properly formatted steps (each in its own tag)
                    proper_steps = re.findall(r'<step>.*?Step\s+\d+:.*?</step>', response_parts[0], re.DOTALL)
                    proper_step_count = len(proper_steps)
                    
                    # Check if each step is in its own tag
                    steps_properly_tagged = proper_step_count == step_count
                    
                    # Check if steps are in order and each appears exactly once
                    if all(count == 1 for count in step_counts.values()) and all(
                        response_parts[0].find(f"Step {i}") < response_parts[0].find(f"Step {i+1}")
                        for i in range(1, step_count)
                    ) and steps_properly_closed and steps_properly_tagged:
                        validation_reward += self.config.solution_ordered_steps_reward
                        self.logger.info(f"Steps are in correct order, unique, and properly closed (+{self.config.solution_ordered_steps_reward})")
                    elif not steps_properly_tagged:
                        self.logger.info(f"Steps are not properly tagged: found {proper_step_count} properly tagged steps out of {step_count} total steps")
                    else:
                        # Log which steps are duplicated
                        duplicates = [i for i, count in step_counts.items() if count > 1]
                        if duplicates:
                            self.logger.info(f"Duplicate steps found: {duplicates}")
                        if not steps_properly_closed:
                            self.logger.info(f"Step tags not properly closed: {opening_tags} opening, {closing_tags} closing")
            
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
            'structure_rewards', 'syntax_rewards', 'execution_rewards', 'correctness_rewards',
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
            
            # 1. Check for thinking and response sections (structure reward)
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if has_thinking and has_response:
                structure_reward = self.config.structure_reward
                reward += structure_reward
                self.stats.reward_components['structure_rewards'] = self.stats.reward_components.get('structure_rewards', 0) + 1
                self.logger.info(f"Applied structure reward: +{structure_reward:.3f}")
            else:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
            
            # Extract code from the completion
            code = extract_code_from_response(completion)
            if not code:
                self.logger.info("No code found in completion")
                return reward
            
            # 2. Check code quality (syntax reward)
            code_quality_passed, quality_message = check_code_quality(code)
            if code_quality_passed:
                syntax_reward = self.config.syntax_reward
                reward += syntax_reward
                self.stats.reward_components['syntax_rewards'] = self.stats.reward_components.get('syntax_rewards', 0) + 1
                self.stats.reward_components['syntax_valid_solutions'] = self.stats.reward_components.get('syntax_valid_solutions', 0) + 1
                self.logger.info(f"Applied syntax reward: +{syntax_reward:.3f}")
            else:
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

class CompletionReward(BaseReward):
    """Reward class for solution completion evaluation"""
    
    __name__ = "completion_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'step_continuity_rewards', 'diversity_bonuses', 'similarity_penalties', 'total_length_penalty', 'correct_answers', 'incorrect_answers', 'total_rewards', 'average_reward'],
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
                
            # Check step continuity and numbering
            # Extract steps from partial solution to determine last step number
            partial_steps = re.findall(r'<step>Step\s+(\d+):', partial_solution, re.IGNORECASE)
            last_partial_step = int(partial_steps[-1]) if partial_steps else 0
            
            # Check if partial solution has properly closed step tags
            opening_tags_partial = len(re.findall(r'<step>', partial_solution, re.IGNORECASE))
            closing_tags_partial = len(re.findall(r'</step>', partial_solution, re.IGNORECASE))
            if opening_tags_partial != closing_tags_partial:
                self.logger.info(f"Partial solution has mismatched step tags: {opening_tags_partial} opening, {closing_tags_partial} closing")
            
            # Extract steps from completion
            completion_steps = re.findall(r'<step>Step\s+(\d+):', completion, re.IGNORECASE)
            
            # Check if completion has properly closed step tags
            opening_tags_completion = len(re.findall(r'<step>', completion, re.IGNORECASE))
            closing_tags_completion = len(re.findall(r'</step>', completion, re.IGNORECASE))
            
            # Extract properly formatted steps (each in its own tag)
            proper_steps = re.findall(r'<step>.*?Step\s+\d+:.*?</step>', completion, re.DOTALL)
            proper_step_count = len(proper_steps)
            
            # Check if each step is in its own tag
            steps_properly_tagged = proper_step_count == len(completion_steps)
            
            if opening_tags_completion != closing_tags_completion:
                self.logger.info(f"Completion has mismatched step tags: {opening_tags_completion} opening, {closing_tags_completion} closing")
                # We don't fail here as the model might be learning to close tags properly
            
            if not steps_properly_tagged:
                self.logger.info(f"Steps are not properly tagged: found {proper_step_count} properly tagged steps out of {len(completion_steps)} total steps")
                # We don't fail here but this will affect the step continuity reward
            
            # Check if completion continues step numbering correctly
            step_continuity_correct = True
            if completion_steps:
                try:
                    first_completion_step = int(completion_steps[0])
                    if first_completion_step != last_partial_step + 1:
                        step_continuity_correct = False
                        self.logger.info(f"Step numbering incorrect: Expected {last_partial_step + 1}, got {first_completion_step}")
                    
                    # Check if steps are in sequence
                    for i in range(1, len(completion_steps)):
                        if int(completion_steps[i]) != int(completion_steps[i-1]) + 1:
                            step_continuity_correct = False
                            self.logger.info(f"Step sequence broken: {completion_steps[i-1]} followed by {completion_steps[i]}")
                            break
                except (ValueError, IndexError) as e:
                    step_continuity_correct = False
                    self.logger.info(f"Error parsing step numbers: {str(e)}")
            else:
                # No steps found in completion
                step_continuity_correct = False
                self.logger.info("No steps found in completion")
            
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




