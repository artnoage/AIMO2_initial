import re
import os
import json
import asyncio
import torch
import logging
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
from transformers import AutoTokenizer, AutoModel
from .config import GRPOConfig
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer, validate_solution

@dataclass 
class RewardConfig:
    """Base configuration for reward calculation"""
    model_type: str
    numeric_tolerance: float = 1e-6
    logging_dir: str = "logs"
    stats_dir: str = "statistics"

class RewardStats:
    """Base class for tracking reward statistics"""
    def __init__(self, config: RewardConfig):
        self.config = config
        self.total_batches = 0
        self.total_rewards = 0
        self.reward_distribution = {}
        self.start_time = datetime.now()
        
    def update(self, rewards: List[float], **kwargs):
        """Update statistics with new rewards"""
        self.total_batches += 1
        for r in rewards:
            self.total_rewards += r
            r_rounded = round(r, 6)
            self.reward_distribution[r_rounded] = self.reward_distribution.get(r_rounded, 0) + 1
            
    def save_statistics(self, output_dir: str):
        """Save current statistics to JSON"""
        stats_dir = Path(output_dir) / self.config.stats_dir
        stats_dir.mkdir(exist_ok=True)
        
        stats = {
            'total_batches': self.total_batches,
            'total_rewards': self.total_rewards,
            'reward_distribution': {str(k): v for k, v in self.reward_distribution.items()},
            'training_duration': str(datetime.now() - self.start_time)
        }
        
        stats_file = stats_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def get_summary(self) -> str:
        """Get a human-readable summary of statistics"""
        total_samples = sum(self.reward_distribution.values())
        if total_samples == 0:
            return "No samples processed yet"
            
        elapsed = datetime.now() - self.start_time
        avg_reward = self.total_rewards / total_samples if total_samples > 0 else 0
        
        return (
            f"Training time: {elapsed}\n"
            f"Processed {self.total_batches} batches\n"
            f"Average reward: {avg_reward:.6f}\n"
            f"Total samples: {total_samples}"
        )

class BaseReward:
    """Base class for reward calculation"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.stats = RewardStats(config)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.config.logging_dir) / self.config.model_type
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger(f'reward_{self.config.model_type}')
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
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion"""
        raise NotImplementedError("Subclasses must implement calculate_reward")
        
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Calculate rewards for a batch of completions"""
        rewards = [self.calculate_reward(comp, **kwargs) for comp in completions]
        self.stats.update(rewards, **kwargs)
        return rewards

    async def calculate_reward_async(self, completion: str, **kwargs) -> float:
        """Async version of calculate_reward"""
        return self.calculate_reward(completion, **kwargs)
        
    async def __call_async__(self, completions: List[str], **kwargs) -> List[float]:
        """Async version of __call__"""
        rewards = await asyncio.gather(*[
            self.calculate_reward_async(comp, **kwargs) 
            for comp in completions
        ])
        self.stats.update(rewards, **kwargs)
        return rewards

class SolutionReward(BaseReward):
    """Reward class for basic solution evaluation"""
    
    def __init__(self, config: GRPOConfig):
        super().__init__(config)
        
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
            reward += self.config.solution_base_reward
            
        # Add validation reward
        is_valid, _ = validate_solution(completion)
        if is_valid:
            reward += self.config.solution_validation_reward
            
        # Apply length penalty
        length_penalty = len(completion) * self.config.solution_length_penalty_factor
        reward -= length_penalty
        
        # Update statistics
        self.stats.reward_components = getattr(self.stats, 'reward_components', {})
        self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + (1 if reward >= self.config.solution_base_reward else 0)
        self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + (1 if is_valid else 0)
        self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
        
        return reward

class GroupReward(BaseReward):
    """Reward class for group-based solution evaluation"""
    
    def __init__(self, config: GRPOConfig, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__(config)
        # Initialize embedding model for similarity checking
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze embedding model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for a list of texts"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return F.normalize(embeddings, p=2, dim=1)
            
    def compute_similarity_matrix(self, solutions: List[str]) -> torch.Tensor:
        """Compute pairwise similarities between solutions"""
        embeddings = self.get_embeddings(solutions)
        return torch.mm(embeddings, embeddings.t())
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion within its group context"""
        group = kwargs.get('group', {})
        if not group:
            self.logger.warning("No group context provided")
            return 0.0
            
        # Extract group information
        completions = group.get('completions', [])
        correct_answer = group.get('correct_answer')
        group_index = group.get('index', 0)
        
        if not completions or not correct_answer:
            return 0.0
            
        # Calculate correctness for all completions
        results = []
        for comp in completions:
            model_answer = extract_answer_from_solution(comp)
            if model_answer is None:
                results.append(False)
                continue
                
            model_numeric, _ = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(correct_answer)
            
            if model_numeric is None or correct_numeric is None:
                results.append(False)
                continue
                
            results.append(abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance)
            
        # Calculate similarity matrix
        similarity_matrix = self.compute_similarity_matrix(completions)
        
        # Calculate reward components
        is_correct = results[group_index]
        reward = self.config.group_base_reward if is_correct else 0.0
        
        # Majority bonus
        correct_count = sum(results)
        is_in_majority = (is_correct and correct_count > len(completions) / 2) or \
                        (not is_correct and (len(completions) - correct_count) > len(completions) / 2)
        if is_in_majority:
            reward += self.config.group_majority_bonus if is_correct else self.config.group_majority_bonus * 0.1
            
        # Diversity bonus
        similarities = similarity_matrix[group_index]
        similarities[group_index] = 0  # Remove self-similarity
        avg_similarity = similarities.mean().item()
        
        if avg_similarity < self.config.group_similarity_threshold_low:  # Unique solution
            reward += self.config.group_diversity_bonus if is_correct else self.config.group_diversity_bonus * 0.1
        elif avg_similarity > self.config.group_similarity_threshold_high:  # Very similar to others
            reward -= self.config.group_diversity_bonus / 2 if is_correct else self.config.group_diversity_bonus * 0.05
            
        # Update statistics
        self.stats.reward_components = getattr(self.stats, 'reward_components', {})
        self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + (1 if is_correct else 0)
        self.stats.reward_components['majority_bonuses'] = self.stats.reward_components.get('majority_bonuses', 0) + (1 if is_in_majority else 0)
        self.stats.reward_components['diversity_bonuses'] = self.stats.reward_components.get('diversity_bonuses', 0) + (1 if avg_similarity < self.config.group_similarity_threshold_low else 0)
        
        return reward

class TutorReward(BaseReward):
    """Reward class for tutor response evaluation"""
    
    def __init__(self, config: GRPOConfig):
        super().__init__(config)
        
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
            reward = self.config.tutor_structure_base_reward
            if substitution:
                reward -= self.config.tutor_redundant_substitution_penalty
        elif verdict.startswith("Step "):
            try:
                step_num = int(verdict.split()[1])
                if step_num < 0:
                    return 0.0
            except (ValueError, IndexError):
                return 0.0
                
            if not substitution:
                return 0.0
                
            reward = self.config.tutor_structure_base_reward
        else:
            return 0.0
            
        # Analysis reward
        if analysis:
            length_penalty = len(analysis) * self.config.tutor_analysis_length_cost
            reward += self.config.tutor_analysis_reward - length_penalty
            
        # Process step verdict
        if verdict.startswith("Step "):
            substitution_steps = self.split_into_steps(substitution)
            if len(substitution_steps) > 1:
                reward -= self.config.tutor_multiple_step_penalty
            else:
                reward += self.config.tutor_single_step_bonus
                
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
                                return self.config.tutor_full_reward
                        else:
                            reward -= self.config.tutor_wrong_boxed_answer_penalty
                            
            length_penalty = len(substitution) * self.config.tutor_substitution_length_cost
            reward += self.config.tutor_substitution_reward - length_penalty
        else:
            reward += self.config.tutor_substitution_reward
            
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
