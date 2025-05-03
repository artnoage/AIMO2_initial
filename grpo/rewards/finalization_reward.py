import re
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import os, sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Union
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.solution_utils import (
    extract_numeric_answer, extract_answer_from_solution, validate_solution
)
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from grpo.rewards.base_reward import BaseReward

class FinalizationReward(BaseReward):
    """Reward class for finalization step evaluation"""

    __name__ = "finalization_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'validation_rewards', 'finalization_rewards'],
        'group_stats': [
            'correct_answers', 'incorrect_answers', 'finalized_solutions'
        ],
        'plurality_stats': [
            'plurality_correct_rate', 'avg_plurality_percentage', 'avg_completion_length'
        ],
        'finalization_stats': [
            'total_finalizations', 'correct_finalizations', 'incorrect_finalizations',
            'finalization_accuracy'
        ]
    }

    def __init__(self, config: RewardConfig):
        super().__init__(config)

        # Numerical tolerance for grouping similar answers
        self.answer_grouping_tolerance = 1e-2

        # Initialize finalization-specific stats
        self.finalization_stats = {
            'total_finalizations': 0,
            'correct_finalizations': 0,
            'incorrect_finalizations': 0,
            'finalization_accuracy': 0.0
        }

    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion with finalization step"""
        try:
            # Get context from kwargs
            correct_answer = kwargs.get('answer')
            batch_index = kwargs.get('reward_index', len(self.stats.current_batch['answers']) if hasattr(self.stats, 'current_batch') else 0)
            reward = 0.0

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

            if not correct_answer:
                self.logger.warning("Missing required correct answer")

                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)

                # Store empty results
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = 0.0
                self.stats.current_batch['code_lengths'][batch_index] = 0
                self.stats.current_batch['completions'][batch_index] = completion

                return 0.0

            self.logger.info(f"Processing finalization completion")

            # Structure validation rewards
            has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            has_finalization = bool(re.search(r'<finalization>.*?</finalization>', completion, re.DOTALL))

            if not has_think or not has_response or not has_finalization:
                self.logger.debug(f"Missing required sections - think: {has_think}, response: {has_response}, finalization: {has_finalization}")
                return reward

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

            validation_reward = 0.0
            
            # Calculate base reward
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            if is_correct:
                base_reward = self.config.base_reward
                reward += base_reward
                self.logger.info(f"Applied base reward: +{base_reward:.3f}")
                self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + 1
                self.stats.reward_components['correct_answers'] = self.stats.reward_components.get('correct_answers', 0) + 1
                self.stats.group_stats['correct_answers'] = self.stats.group_stats.get('correct_answers', 0) + 1
            else:
                self.stats.reward_components['incorrect_answers'] = self.stats.reward_components.get('incorrect_answers', 0) + 1
                self.stats.group_stats['incorrect_answers'] = self.stats.group_stats.get('incorrect_answers', 0) + 1

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
                self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + 1
                self.logger.info(f"Applied total validation reward: +{validation_reward:.3f}")

            # Extract finalization content to check if model finalizes the answer
            finalization_match = re.search(r'<finalization>(.*?)</finalization>', completion, re.DOTALL)
            finalization_content = finalization_match.group(1) if finalization_match else ""

            # Check if finalization is present and contains the answer
            has_finalized_answer = False
            if finalization_content:
                # Check if the finalization contains the answer
                finalization_answer = extract_answer_from_solution(finalization_content)
                if finalization_answer is not None:
                    finalization_numeric, _ = extract_numeric_answer(finalization_answer)
                    if finalization_numeric is not None:
                        has_finalized_answer = abs(finalization_numeric - model_numeric) <= self.config.numeric_tolerance
                        self.logger.info(f"Finalization answer: {finalization_numeric}, model answer: {model_numeric}, match: {has_finalized_answer}")

            # Update finalization statistics
            self.finalization_stats['total_finalizations'] += 1

            # Initialize finalization reward
            finalization_reward = 0.0

            # Apply finalization reward if the answer is correct and finalized
            if is_correct and has_finalized_answer:
                finalization_reward = 0.5
                self.logger.info(f"Applied finalization reward: +{finalization_reward:.1f}")
                self.stats.reward_components['finalization_rewards'] = self.stats.reward_components.get('finalization_rewards', 0) + 1
                self.finalization_stats['correct_finalizations'] += 1
                self.stats.group_stats['finalized_solutions'] = self.stats.group_stats.get('finalized_solutions', 0) + 1
                reward += finalization_reward
            elif not is_correct and has_finalized_answer:
                self.logger.info(f"Answer is incorrect but model finalized it")
                self.finalization_stats['incorrect_finalizations'] += 1

            # Update finalization accuracy statistics
            total_finalizations = self.finalization_stats['total_finalizations']
            correct_finalizations = self.finalization_stats['correct_finalizations']
            if total_finalizations > 0:
                self.finalization_stats['finalization_accuracy'] = correct_finalizations / total_finalizations

            # Log finalization statistics
            self.logger.info(f"Finalization stats - Accuracy: {self.finalization_stats['finalization_accuracy']:.2%}, "
                            f"Correct finalizations: {correct_finalizations}/{total_finalizations}")

            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.reward_components.get('correct_answers', 0) + self.stats.reward_components.get('incorrect_answers', 0)
            self.stats.reward_components['average_reward'] = \
                self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)

            return reward

        except Exception as e:
            self.logger.error(f"Error calculating finalization reward: {str(e)}")
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
