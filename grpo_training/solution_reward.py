from typing import List, Dict, Optional, Tuple
from .reward_base import BaseReward, RewardConfig
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer, validate_solution

class SolutionReward(BaseReward):
    """Reward class for basic solution evaluation"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        # Additional reward settings
        self.base_reward = 2.0  # Reward for correct answer
        self.validation_reward = 0.2  # Reward for valid solution format
        self.length_penalty_factor = 0.0001  # Per character penalty
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion"""
        reward = 0.0
        
        # Extract and validate the answer
        model_answer = extract_answer_from_solution(completion)
        if model_answer is None:
            self.logger.debug("No boxed answer found")
            return reward
            
        # Convert to numeric values
        model_numeric, debug_info = extract_numeric_answer(model_answer)
        correct_answer = kwargs.get('correct_answer')
        if not correct_answer:
            self.logger.warning("No correct answer provided")
            return reward
            
        correct_numeric, _ = extract_numeric_answer(correct_answer)
        
        if model_numeric is None or correct_numeric is None:
            self.logger.debug(f"Could not extract numeric values - Model: {model_answer}, Correct: {correct_answer}")
            return reward
            
        # Check correctness
        if abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance:
            reward += self.base_reward
            
        # Add validation reward
        is_valid, _ = validate_solution(completion)
        if is_valid:
            reward += self.validation_reward
            
        # Apply length penalty
        length_penalty = len(completion) * self.length_penalty_factor
        reward -= length_penalty
        
        # Update statistics
        self.stats.reward_components = getattr(self.stats, 'reward_components', {})
        self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + (1 if reward >= self.base_reward else 0)
        self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + (1 if is_valid else 0)
        self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
        
        return reward
