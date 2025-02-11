import re
import asyncio
from typing import List, Dict, Optional, Tuple
from .reward_base import BaseReward, RewardConfig
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer

class TutorReward(BaseReward):
    """Reward class for tutor response evaluation"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        # Reward settings
        self.structure_base_reward = 0.2
        self.analysis_reward = 0.2
        self.substitution_reward = 0.4
        self.single_step_bonus = 0.2
        self.multiple_step_penalty = 0.4
        self.full_reward = 5.0
        
        # Penalty settings
        self.analysis_length_cost = 0.0001
        self.substitution_length_cost = 0.0001
        self.redundant_substitution_penalty = 0.1
        self.wrong_boxed_answer_penalty = 1.0
        
    def extract_sections(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract Analysis, Verdict and Substitution sections"""
        analysis_match = re.search(r'</Analysis>\s*(.*?)\s*<Analysis>', response, re.DOTALL)
        verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
        substitution_match = re.search(r'</Substitution>\s*(.*?)\s*<Substitution>', response, re.DOTALL)
        
        return (
            analysis_match.group(1).strip() if analysis_match else None,
            verdict_match.group(1).strip() if verdict_match else None,
            substitution_match.group(1).strip() if substitution_match else None
        )
        
    def split_into_steps(self, solution: str) -> List[str]:
        """Split solution into analysis and numbered steps"""
        parts = solution.split("Step")
        if not parts:
            return []
            
        steps = []
        if "analysis" in parts[0].lower():
            steps.append(parts[0].strip())
            
        for step in parts[1:]:
            if step.strip():
                steps.append(("Step" + step).strip())
                
        return steps
        
    async def calculate_reward_async(self, completion: str, **kwargs) -> float:
        """Async version of reward calculation"""
        # Extract sections
        analysis, verdict, substitution = self.extract_sections(completion)
        
        if verdict is None:
            self.logger.debug(f"Missing verdict section in completion: {completion[:100]}...")
            return 0.0
            
        polar_verdicts = ["The answer is correct", "The whole approach is wrong"]
        reward = 0.0
        
        # Basic structure reward
        if verdict in polar_verdicts:
            reward = self.structure_base_reward
            if substitution:
                reward -= self.redundant_substitution_penalty
        elif verdict.startswith("Step "):
            try:
                step_num = int(verdict.split()[1])
                if step_num < 0:
                    return 0.0
            except (ValueError, IndexError):
                return 0.0
                
            if not substitution:
                return 0.0
                
            reward = self.structure_base_reward
        else:
            return 0.0
            
        # Analysis reward
        if analysis:
            length_penalty = len(analysis) * self.analysis_length_cost
            reward += self.analysis_reward - length_penalty
            
        # Process step verdict
        if verdict.startswith("Step "):
            substitution_steps = self.split_into_steps(substitution)
            if len(substitution_steps) > 1:
                reward -= self.multiple_step_penalty
            else:
                reward += self.single_step_bonus
                
            # Check substitution answer
            boxed_answer = extract_answer_from_solution(substitution)
            if boxed_answer:
                numeric_value, _ = extract_numeric_answer(boxed_answer)
                correct_answer = kwargs.get('correct_answer')
                if correct_answer:
                    correct_numeric, _ = extract_numeric_answer(correct_answer)
                    if numeric_value is not None and correct_numeric is not None:
                        if abs(numeric_value - correct_numeric) <= self.config.numeric_tolerance:
                            solution = kwargs.get('solution', '')
                            solution_steps = self.split_into_steps(solution)
                            if step_num == len(solution_steps) - 1:
                                return self.full_reward
                        else:
                            reward -= self.wrong_boxed_answer_penalty
                            
            length_penalty = len(substitution) * self.substitution_length_cost
            reward += self.substitution_reward - length_penalty
        else:
            reward += self.substitution_reward
            
        # Update statistics
        self.stats.reward_components = getattr(self.stats, 'reward_components', {})
        self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + 1
        if analysis:
            self.stats.reward_components['analysis_rewards'] = self.stats.reward_components.get('analysis_rewards', 0) + 1
        if substitution:
            self.stats.reward_components['substitution_rewards'] = self.stats.reward_components.get('substitution_rewards', 0) + 1
            
        return reward
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Synchronous wrapper for async reward calculation"""
        return asyncio.run(self.calculate_reward_async(completion, **kwargs))
