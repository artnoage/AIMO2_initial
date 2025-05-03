import re
import asyncio
import time
import logging
from datetime import datetime
from pathlib import Path
import os, sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Union
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.solution_utils import (
    extract_numeric_answer, extract_answer_from_solution, extract_code_from_response,
    check_code_quality, run_code_safely
)
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from grpo.rewards.base_reward import BaseReward

class ProgrammingReward(BaseReward):
    """Reward class for programming solution evaluation"""

    __name__ = "programming_reward"
    relevant_stats = {
        'reward_components': [
            'syntax_rewards', 'execution_rewards', 'correctness_rewards',
            'correct_solutions', 'syntax_valid_solutions',
            'execution_valid_solutions', 'total_rewards', 'average_reward'
        ],
        'programming_stats': [
            'correct_solutions', 'incorrect_solutions', 'syntax_errors',
            'execution_errors', 'timeout_errors'
        ],
        'plurality_stats': [
            'plurality_correct_rate', 'avg_plurality_percentage', 'avg_completion_length'
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

        # Numerical tolerance for grouping similar answers
        self.answer_grouping_tolerance = 1e-2

    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a programming solution"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            batch_index = kwargs.get('reward_index', len(self.stats.current_batch['answers']))

            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for programming reward calculation")
                return 0.0

            # Initialize reward
            reward = 0.0

            # Initialize tracking variables for this completion
            model_answer = None
            model_numeric = None
            is_correct = False
            execution_time = 0.0
            code_length = 0

            # 1. Check for think and response sections
            has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))

            if not has_think or not has_response:
                self.logger.info(f"Missing {'think' if not has_think else ''} {'response' if not has_response else ''} section(s)")
                return 0.0

            # Check for glimpses of reasoning in think section
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
            has_glimpses = False
            if think_match:
                has_glimpses = True # Simple check for presence of think content


            # Extract code from the completion
            # First check if response section exists
            if not has_response:
                self.logger.info("No response section found in completion")
                # We can still try to extract code from the whole completion
                code = extract_code_from_response(completion)
                if not code:
                    self.logger.info("No code found in completion")

                    # Ensure lists are long enough for this batch index
                    self._ensure_batch_lists_length(batch_index)

                    # Store empty results
                    self.stats.current_batch['answers'][batch_index] = None
                    self.stats.current_batch['is_correct'][batch_index] = False
                    self.stats.current_batch['execution_times'][batch_index] = 0.0
                    self.stats.current_batch['code_lengths'][batch_index] = 0
                    self.stats.current_batch['completions'][batch_index] = completion

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

                        # Ensure lists are long enough for this batch index
                        self._ensure_batch_lists_length(batch_index)

                        # Store empty results
                        self.stats.current_batch['answers'][batch_index] = None
                        self.stats.current_batch['is_correct'][batch_index] = False
                        self.stats.current_batch['execution_times'][batch_index] = 0.0
                        self.stats.current_batch['code_lengths'][batch_index] = 0
                        self.stats.current_batch['completions'][batch_index] = completion

                        return reward

            self.logger.info(f"Extracted code length: {len(code)} characters")
            code_length = len(code)

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
                self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0) + reward
                total_samples = self.stats.total_batches
                self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0) / max(1, total_samples)
                return reward  # Return early if syntax is invalid

            # 3. Run the code and check if it produces a valid output (execution reward)
            start_time = time.time()
            execution_success, result, error_message = run_code_safely(code, timeout=self.config.timeout)
            execution_time = time.time() - start_time

            if execution_success and result is not None:
                execution_reward = self.config.execution_reward
                reward += execution_reward
                self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                self.stats.reward_components['execution_valid_solutions'] = self.stats.reward_components.get('execution_valid_solutions', 0) + 1
                self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
                model_numeric = result
            else:
                self.logger.info(f"Code execution failed: {error_message}")
                if "timed out" in error_message:
                    self.stats.programming_stats['timeout_errors'] += 1
                else:
                    self.stats.programming_stats['execution_errors'] += 1

                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)

                # Store results even for failed execution
                self.stats.current_batch['answers'][batch_index] = None
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = execution_time
                self.stats.current_batch['code_lengths'][batch_index] = code_length
                self.stats.current_batch['completions'][batch_index] = completion

                # Update total rewards and average before returning
                self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0) + reward
                total_samples = self.stats.total_batches
                self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0) / max(1, total_samples)
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

                # Ensure lists are long enough for this batch index
                self._ensure_batch_lists_length(batch_index)

                # Store results even for failed conversion
                self.stats.current_batch['answers'][batch_index] = model_numeric
                self.stats.current_batch['is_correct'][batch_index] = False
                self.stats.current_batch['execution_times'][batch_index] = execution_time
                self.stats.current_batch['code_lengths'][batch_index] = code_length
                self.stats.current_batch['completions'][batch_index] = completion

                return reward

            # Compare with tolerance
            is_correct = abs(correct_answer - model_numeric) <= self.config.numeric_tolerance
            if is_correct:
                correctness_reward = self.config.correctness_reward

                # Apply bonus for glimpses of reasoning
                if has_glimpses:
                    correctness_reward *= 3
                    self.logger.info(f"Applied 3x bonus for glimpses of reasoning")

                reward += correctness_reward
                self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                self.stats.reward_components['correct_solutions'] = self.stats.reward_components.get('correct_solutions', 0) + 1
                self.stats.programming_stats['correct_solutions'] += 1
                self.logger.info(f"Applied correctness reward: +{correctness_reward:.3f}")
            else:
                self.stats.programming_stats['incorrect_solutions'] += 1
                self.logger.info(f"Incorrect answer: expected {correct_answer}, got {model_numeric}")

            # Ensure lists are long enough for this batch index
            self._ensure_batch_lists_length(batch_index)

            # Store the results for this completion
            self.stats.current_batch['answers'][batch_index] = model_numeric
            self.stats.current_batch['is_correct'][batch_index] = is_correct
            self.stats.current_batch['execution_times'][batch_index] = execution_time
            self.stats.current_batch['code_lengths'][batch_index] = code_length
            self.stats.current_batch['completions'][batch_index] = completion

            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)

            return reward

        except Exception as e:
            self.logger.error(f"Error calculating programming reward: {str(e)}")
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
