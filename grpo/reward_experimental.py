import re
import asyncio
import torch
import logging
from datetime import datetime
from pathlib import Path
import os, sys
from collections import defaultdict
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
            
        # Reset current batch data for this new batch
        self.stats.current_batch = {
            'answers': [],
            'is_correct': [],
            'execution_times': [],
            'code_lengths': [],
            'completions': []
        }
            
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
        
     
        self.logger.info(f"Rewards before: {rewards}")
            
           
        
        # Update stats and print batch summary
        self.stats.update(rewards, completions=completions, example_type=kwargs.get('example_type', []))
        
        # Process the batch results to calculate plurality metrics
        self._finalize_batch()
        
        # Print reward-specific statistics summary every batch
        self.logger.info("\nReward Statistics Summary:")
        self.logger.info(self.stats.get_summary(getattr(self, 'relevant_stats', None)))
        
        return rewards
        
    def _finalize_batch(self):
        """Calculate batch-level statistics including plurality voting with numerical grouping"""
        # Skip if no results
        if not self.stats.current_batch['answers']:
            return
        
        # Filter out None values
        valid_results = [(ans, correct) for ans, correct in 
                         zip(self.stats.current_batch['answers'], 
                             self.stats.current_batch['is_correct']) 
                         if ans is not None]
        
        if not valid_results:
            return
        
        # Group similar answers using the tolerance
        grouped_answers = defaultdict(list)
        
        for idx, (ans, correct) in enumerate(valid_results):
            # Find if this answer belongs to an existing group
            found_group = False
            for group_key in grouped_answers:
                if abs(ans - group_key) <= self.answer_grouping_tolerance:
                    # Add to existing group
                    grouped_answers[group_key].append((idx, ans, correct))
                    found_group = True
                    break
            
            if not found_group:
                # Create new group
                grouped_answers[ans].append((idx, ans, correct))
        
        # Find the plurality winner (most common answer group)
        if grouped_answers:
            # Get the group with the most answers
            plurality_group, plurality_indices = max(grouped_answers.items(), 
                                                    key=lambda x: len(x[1]))
            
            # Calculate what percentage of valid answers this represents
            plurality_count = len(plurality_indices)
            plurality_percentage = plurality_count / len(valid_results)
            
            # Check if the plurality answer is correct
            plurality_correct = any(correct for _, _, correct in plurality_indices)
            
            # Calculate average completion length
            avg_completion_length = sum(len(comp) for comp in self.stats.current_batch['completions'] if comp) / len(self.stats.current_batch['completions']) if self.stats.current_batch['completions'] else 0
            
            # Store batch results
            batch_result = {
                'plurality_answer': plurality_group,
                'plurality_count': plurality_count,
                'plurality_percentage': plurality_percentage,
                'plurality_correct': plurality_correct,
                'total_answers': len(valid_results),
                'correct_answers': sum(1 for _, _, correct in plurality_indices if correct),
                'avg_code_length': 0,  # Not applicable for solution reward
                'avg_execution_time': 0,  # Not applicable for solution reward
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to batch history
            if not hasattr(self.stats, 'batch_results'):
                self.stats.batch_results = []
            self.stats.batch_results.append(batch_result)
            
            # Update plurality statistics
            if not hasattr(self.stats, 'plurality_stats'):
                self.stats.plurality_stats = {
                    'plurality_correct_count': 0,
                    'total_batches': 0,
                    'plurality_correct_rate': 0.0,
                    'avg_plurality_percentage': 0.0,
                    'avg_completion_length': 0.0
                }
                
            self.stats.plurality_stats['plurality_correct_count'] += int(plurality_correct)
            self.stats.plurality_stats['total_batches'] += 1
            
            if self.stats.plurality_stats['total_batches'] > 0:
                self.stats.plurality_stats['plurality_correct_rate'] = (
                    self.stats.plurality_stats['plurality_correct_count'] / 
                    self.stats.plurality_stats['total_batches']
                )
            
            # Update average plurality percentage (how dominant is the most common answer)
            prev_avg = self.stats.plurality_stats['avg_plurality_percentage']
            prev_batches = self.stats.plurality_stats['total_batches'] - 1
            
            if prev_batches > 0:
                self.stats.plurality_stats['avg_plurality_percentage'] = (
                    (prev_avg * prev_batches + plurality_percentage) / 
                    self.stats.plurality_stats['total_batches']
                )
            else:
                self.stats.plurality_stats['avg_plurality_percentage'] = plurality_percentage
            
            # Update average completion length
            prev_avg_length = self.stats.plurality_stats['avg_completion_length']
            
            if prev_batches > 0:
                self.stats.plurality_stats['avg_completion_length'] = (
                    (prev_avg_length * prev_batches + avg_completion_length) / 
                    self.stats.plurality_stats['total_batches']
                )
            else:
                self.stats.plurality_stats['avg_completion_length'] = avg_completion_length
            
            # Log the results
            self.logger.info(
                f"Batch plurality results: answer={plurality_group}, " +
                f"count={plurality_count}/{len(valid_results)} ({plurality_percentage:.2%}), " +
                f"correct={plurality_correct}, " +
                f"overall rate={self.stats.plurality_stats['plurality_correct_rate']:.2%}"
            )
            
            # Log the answer groups for debugging
            group_info = []
            for group_key, indices in grouped_answers.items():
                correct_in_group = any(correct for _, _, correct in indices)
                group_info.append(f"{group_key}: {len(indices)} answers, correct={correct_in_group}")
            
            self.logger.info(f"Answer groups (tolerance={self.answer_grouping_tolerance}):")
            for info in group_info:
                self.logger.info(f"  {info}")


class SolutionReward(BaseReward):
    """Reward class for group-based solution evaluation"""
    
    __name__ = "solution_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'validation_rewards', 'verification_rewards'],
        'group_stats': [
            'correct_answers', 'incorrect_answers', 'verified_solutions'
        ],
        'plurality_stats': [
            'plurality_correct_rate', 'avg_plurality_percentage', 'avg_completion_length',
            'batch_plurality_correct', 'batch_plurality_percentage', 'batch_total_answers',
            'batch_correct_answers', 'batch_correct_rate'
        ]
    }
    
    def __init__(self, config: RewardConfig, similarity_checker=None):
        super().__init__(config)
        
        # Numerical tolerance for grouping similar answers
        self.answer_grouping_tolerance = 1e-2
        
        # Initialize verification-specific stats
        if not hasattr(self.stats.group_stats, 'verified_solutions'):
            self.stats.group_stats['verified_solutions'] = 0
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion with group context"""
        try:
            # Get group context from kwargs
            group_completions = kwargs.get('group_completions', [])
            group_answers = kwargs.get('group_answers', [])
            group_indices = kwargs.get('group_indices', [])
            group_idx = kwargs.get('group_idx', 0)
            correct_answer = kwargs.get('answer')
            batch_index = kwargs.get('reward_index', len(self.stats.current_batch['answers']) if hasattr(self.stats, 'current_batch') else 0)
            reward=0.0
            
            # Initialize tracking variables for this completion
            model_answer = None
            model_numeric = None
            is_correct = False
            
            # Ensure current_batch exists in stats
            if not hasattr(self.stats, 'current_batch'):
                self.stats.current_batch = {
                    'answers': [],
                    'is_correct': [],
                    'execution_times': [],
                    'code_lengths': [],
                    'completions': []
                }
            
            if not all([group_completions, group_answers, group_indices]):
                self.logger.warning(f"Missing required group context - completions: {bool(group_completions)}, answers: {bool(group_answers)}, indices: {bool(group_indices)}")
                
                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)
                
                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion
                
                return 0.0

            self.logger.info(f"Processing completion {group_idx+1}/{len(group_completions)} in group")
            
            # Check for glimpses of reasoning in thinking section
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', completion, re.DOTALL)
            has_glimpses = False
            if thinking_match:
                thinking_content = thinking_match.group(1)
                # Check if any of the glimpses of reasoning are in the thinking content
                from grpo.terms import Glimpses_of_reasoning
                for glimpse in Glimpses_of_reasoning:
                    if glimpse.lower() in thinking_content.lower():
                        has_glimpses = True
                        self.logger.info(f"Found glimpse of reasoning: '{glimpse}'")
                        break
            
            
            # Extract and validate the answer
            model_answer = extract_answer_from_solution(completion)
            if model_answer is None:
                self.logger.info("There is no model_answer")
                
                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)
                
                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion
                
                return reward
                
            # Convert to numeric values
            model_numeric, debug_info = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(str(correct_answer))
            if model_numeric is None or correct_numeric is None:
                self.logger.debug("Could not extract numeric values - returning 0.0")
                
                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)
                
                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion
                
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
                base_reward = self.config.base_reward
                
                # Apply bonus for glimpses of reasoning
                if has_glimpses:
                    base_reward *= 3
                    self.logger.info(f"Applied 3x bonus for glimpses of reasoning")
                
                reward += base_reward
                self.logger.info(f"Applied base reward: +{base_reward:.3f}")
                self.stats.reward_components['base_rewards'] += 1
                self.stats.reward_components['correct_answers'] += 1
                
            else:
                self.stats.reward_components['incorrect_answers'] += 1
                
            # Ensure lists are long enough for this batch index
            self._ensure_batch_lists_length(batch_index)
            
            # Store the results for this completion
            self.stats.current_batch['answers'][batch_index] = model_numeric
            self.stats.current_batch['is_correct'][batch_index] = is_correct
            self.stats.current_batch['execution_times'][batch_index] = 0.0  # Not applicable for solution reward
            self.stats.current_batch['code_lengths'][batch_index] = len(completion)
            self.stats.current_batch['completions'][batch_index] = completion


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
                

            # Update total rewards and average
            self.stats.reward_components['total_rewards'] += reward
            total_samples = self.stats.reward_components['correct_answers'] + self.stats.reward_components['incorrect_answers']
            self.stats.reward_components['average_reward'] = \
                self.stats.reward_components['total_rewards'] / max(1, total_samples)
            
            # If the solution is correct, verify it with the verification agent
            if is_correct:
                # Only verify solutions that have passed basic correctness checks
                verification_passed = await self.verify_solution(
                    problem=kwargs.get('problem', ''),
                    solution=completion,
                    correct_answer=correct_numeric
                )
                
                if verification_passed:
                    # Apply verification reward
                    verification_reward = self.config.verification_reward
                    reward += verification_reward
                    self.logger.info(f"Applied verification reward: +{verification_reward:.3f}")
                    
                    # Update verification stats
                    self.stats.reward_components['verification_rewards'] = self.stats.reward_components.get('verification_rewards', 0) + 1
                    self.stats.group_stats['verified_solutions'] += 1
                    
                    # Update total rewards again after verification
                    self.stats.reward_components['total_rewards'] += verification_reward
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
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Ensure lists are long enough for this batch index
            self._ensure_batch_lists_length(batch_index)
            
            # Store empty results in case of exception
            self.stats.current_batch['answers'][batch_index] = None
            self.stats.current_batch['is_correct'][batch_index] = False
            self.stats.current_batch['execution_times'][batch_index] = 0.0
            self.stats.current_batch['code_lengths'][batch_index] = 0
            self.stats.current_batch['completions'][batch_index] = completion
            
            return 0.0
    
    async def verify_solution(self, problem: str, solution: str, correct_answer: float) -> bool:
        """
        Use a verification agent to check if the solution is correct.
        This is a placeholder method that would normally call an external agent.
        
        Args:
            problem: The problem statement
            solution: The solution to verify
            correct_answer: The expected answer
            
        Returns:
            bool: True if the solution is verified as correct, False otherwise
        """
        try:
            self.logger.info("Calling verification agent to verify solution")
            
            # Get the model using the benchmark config
            verification_model = get_model(self.config, role="auxiliary")
            
            # In a real implementation, we would create a verification agent and call it
            # For now, we'll simulate a verification process with a placeholder
            
            # Extract the answer from the solution
            model_answer = extract_answer_from_solution(solution)
            if model_answer is None:
                self.logger.info("Verification failed: No answer found in solution")
                return False
                
            # Convert to numeric value
            model_numeric, _ = extract_numeric_answer(model_answer)
            if model_numeric is None:
                self.logger.info("Verification failed: Could not extract numeric answer")
                return False
                
            # Check if the answer is correct (this is a simplified verification)
            # In a real implementation, the verification agent would analyze the solution steps
            is_correct = abs(model_numeric - correct_answer) <= self.config.numeric_tolerance
            
            # Simulate some verification logic based on solution quality
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', solution, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', solution, re.DOTALL))
            
            # Check for step structure in the response
            response_match = re.search(r'<response>(.*?)</response>', solution, re.DOTALL)
            has_steps = False
            if response_match:
                response_content = response_match.group(1)
                has_steps = bool(re.search(r'Step\s+\d+:', response_content, re.IGNORECASE))
            
            # Verification passes if the answer is correct and the solution has proper structure
            verification_passed = is_correct and has_thinking and has_response and has_steps
            
            if verification_passed:
                self.logger.info("Verification passed: Solution is correct and well-structured")
            else:
                self.logger.info(f"Verification failed: correct={is_correct}, thinking={has_thinking}, response={has_response}, steps={has_steps}")
                
            return verification_passed
            
        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _ensure_batch_lists_length(self, index):
        """Ensure all batch lists are long enough to store data at the given index"""
        for key in ['answers', 'is_correct', 'execution_times', 'code_lengths', 'completions']:
            while len(self.stats.current_batch[key]) <= index:
                if key == 'is_correct':
                    self.stats.current_batch[key].append(False)
                elif key in ['execution_times', 'code_lengths']:
                    self.stats.current_batch[key].append(0.0)
                else:
                    self.stats.current_batch[key].append(None)
