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
        
        # Track section-level stats
        self.section_stats = {
            'missing_analysis': 0,
            'missing_verdict': 0,
            'missing_substitution': 0,
            'invalid_step_number': 0,
            'polar_verdict_with_substitution': 0,
            'step_verdict_without_substitution': 0,
            'multiple_steps_in_substitution': 0
        }
        
        # Track reward components
        self.reward_components = {
            'base_rewards': 0,
            'analysis_rewards': 0,
            'substitution_rewards': 0,
            'step_bonuses': 0,
            'step_penalties': 0,
            'total_analysis_length_penalty': 0.0,
            'total_substitution_length_penalty': 0.0,
            'redundant_substitution_penalties': 0,
            'wrong_boxed_answer_penalties': 0,
            'improvement_bonuses': {
                '0.1': 0,  # 10-40% completions
                '0.2': 0,  # 40-70% completions
                '0.3': 0,  # >70% completions
                'total': 0  # Total count of improvement bonuses
            }
        }
        
        # Track group-specific stats
        self.group_stats = {
            'majority_bonuses': 0,
            'diversity_bonuses': 0,
            'unique_solutions': 0,
            'similar_solutions': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'total_similarity': 0.0
        }
        
        # Track full reward reasons
        self.full_reward_reasons = {
            'correct_answer': 0,
            'wrong_approach': 0,
            'step_correction': 0,
            'final_step_correct': 0
        }
        
    def update(self, rewards: List[float], **kwargs):
        """Update statistics with new rewards"""
        self.total_batches += 1
        for r in rewards:
            self.total_rewards += r
            r_rounded = round(r, 6)
            self.reward_distribution[r_rounded] = self.reward_distribution.get(r_rounded, 0) + 1
            
        # Update section stats if provided
        completion = kwargs.get('completion')
        if completion:
            if 'analysis' not in completion.lower():
                self.section_stats['missing_analysis'] += 1
            if 'verdict' not in completion.lower():
                self.section_stats['missing_verdict'] += 1
            if 'substitution' not in completion.lower():
                self.section_stats['missing_substitution'] += 1
                
        # Update group stats if provided
        similarity = kwargs.get('similarity')
        if similarity is not None:
            self.group_stats['total_similarity'] += float(similarity)
            if similarity < 0.7:
                self.group_stats['unique_solutions'] += 1
            elif similarity > 0.9:
                self.group_stats['similar_solutions'] += 1
            
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
        
        # Sort rewards for better readability
        sorted_rewards = sorted(self.reward_distribution.items())
        reward_dist_str = "\n".join(
            f"  {reward:.6f}: {count} samples" 
            for reward, count in sorted_rewards
        )
        
        return (
            f"Training time: {elapsed}\n"
            f"Processed {self.total_batches} batches\n"
            f"Average reward: {avg_reward:.6f}\n"
            f"Total samples: {total_samples}\n"
            f"\nReward Distribution:\n{reward_dist_str}\n"
            f"\nSection Issues:\n"
            f"  Missing analysis: {self.section_stats['missing_analysis']}\n"
            f"  Missing verdict: {self.section_stats['missing_verdict']}\n"
            f"  Step verdict without substitution: {self.section_stats['step_verdict_without_substitution']}\n"
            f"  Polar verdict with substitution: {self.section_stats['polar_verdict_with_substitution']}\n"
            f"  Multiple steps in substitution: {self.section_stats['multiple_steps_in_substitution']}\n"
            f"\nReward Components:\n"
            f"  Base rewards: {self.reward_components['base_rewards']}\n"
            f"  Analysis rewards: {self.reward_components['analysis_rewards']}\n"
            f"  Substitution rewards: {self.reward_components['substitution_rewards']}\n"
            f"  Step bonuses: {self.reward_components['step_bonuses']}\n"
            f"  Step penalties: {self.reward_components['step_penalties']}\n"
            f"\nPenalties:\n"
            f"  Analysis length: {self.reward_components['total_analysis_length_penalty']:.6f}\n"
            f"  Substitution length: {self.reward_components['total_substitution_length_penalty']:.6f}\n"
            f"  Wrong boxed answers: {self.reward_components['wrong_boxed_answer_penalties']}\n"
            f"  Redundant substitutions: {self.reward_components['redundant_substitution_penalties']}\n"
            f"\nGroup Statistics:\n"
            f"  Majority bonuses: {self.group_stats['majority_bonuses']}\n"
            f"  Diversity bonuses: {self.group_stats['diversity_bonuses']}\n"
            f"  Unique solutions: {self.group_stats['unique_solutions']}\n"
            f"  Similar solutions: {self.group_stats['similar_solutions']}\n"
            f"  Average similarity: {self.group_stats['total_similarity']/total_samples if total_samples else 0:.3f}\n"
            f"\nFull Reward Reasons:\n"
            f"  Correct answer: {self.full_reward_reasons['correct_answer']}\n"
            f"  Wrong approach: {self.full_reward_reasons['wrong_approach']}\n"
            f"  Step correction: {self.full_reward_reasons['step_correction']}\n"
            f"  Final step correct: {self.full_reward_reasons['final_step_correct']}"
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
        # Initialize completion agent for validation
        self.completion_agent = CompletionAgent(
            port=config.completion_port,
            model=config.completion_model_name,
            logger=self.logger
        )
        
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
        
    async def calculate_reward_async(self, tutor_response: str, **kwargs) -> float:
        """Calculate reward for a tutor's evaluation of a solution"""
        # Extract sections from tutor's response
        analysis, verdict, substitution = self.extract_sections(tutor_response)
        
        if verdict is None:
            self.logger.debug(f"Missing verdict section in tutor response: {tutor_response[:100]}...")
            return 0.0
            
        # Get problem and student solution from kwargs
        problem = kwargs.get('problem')
        student_solution = kwargs.get('solution')
        correct_answer = kwargs.get('correct_answer')
        
        if not all([problem, student_solution, correct_answer]):
            self.logger.warning("Missing required context (problem, solution, or correct_answer)")
            return 0.0
            
        polar_verdicts = ["The answer is correct", "The whole approach is wrong"]
        reward = 0.0
        
        # Basic structure reward
        if verdict in polar_verdicts:
            reward = self.config.tutor_structure_base_reward
            if substitution:
                reward -= self.config.tutor_redundant_substitution_penalty
                self.stats.reward_components['redundant_substitution_penalties'] += 1
        elif verdict.startswith("Step "):
            try:
                step_num = int(verdict.split()[1])
                if step_num < 0:
                    self.stats.section_stats['invalid_step_number'] += 1
                    return 0.0
            except (ValueError, IndexError):
                self.stats.section_stats['invalid_step_number'] += 1
                return 0.0
                
            if not substitution:
                self.stats.section_stats['step_verdict_without_substitution'] += 1
                return 0.0
                
            reward = self.config.tutor_structure_base_reward
        else:
            return 0.0
            
        # Analysis reward
        if analysis:
            length_penalty = len(analysis) * self.config.tutor_analysis_length_cost
            reward += self.config.tutor_analysis_reward - length_penalty
            self.stats.reward_components['analysis_rewards'] += 1
            self.stats.reward_components['total_analysis_length_penalty'] += length_penalty
            
        # Verify tutor's verdict using completion agent
        if verdict == "The answer is correct":
            # Check if student solution is actually correct
            student_answer = extract_answer_from_solution(student_solution)
            if student_answer:
                student_numeric, _ = extract_numeric_answer(student_answer)
                correct_numeric, _ = extract_numeric_answer(correct_answer)
                if student_numeric is not None and correct_numeric is not None:
                    if abs(student_numeric - correct_numeric) <= self.config.numeric_tolerance:
                        reward = self.config.tutor_full_reward
                        self.stats.full_reward_reasons['correct_answer'] += 1
                    else:
                        # Tutor incorrectly said answer was correct
                        return 0.0
                        
        elif verdict == "The whole approach is wrong":
            if not analysis:
                return reward
                
            # Verify by trying to complete solution from analysis
            try:
                completion = await self.completion_agent.generate(problem, analysis)
                completion_answer = extract_answer_from_solution(completion)
                if completion_answer:
                    completion_numeric, _ = extract_numeric_answer(completion_answer)
                    correct_numeric, _ = extract_numeric_answer(correct_answer)
                    if completion_numeric is not None and correct_numeric is not None:
                        if abs(completion_numeric - correct_numeric) <= self.config.numeric_tolerance:
                            # Tutor incorrectly said approach was wrong
                            return 0.0
                        else:
                            # Tutor correctly identified wrong approach
                            reward = self.config.tutor_full_reward
                            self.stats.full_reward_reasons['wrong_approach'] += 1
            except Exception as e:
                self.logger.warning(f"Error during completion validation: {str(e)}")
                return reward
                
        elif verdict.startswith("Step "):
            solution_steps = self.split_into_steps(student_solution)
            if step_num >= len(solution_steps):
                return reward
                
            # Check if substitution has multiple steps
            substitution_steps = self.split_into_steps(substitution)
            if len(substitution_steps) > 1:
                reward -= self.config.tutor_multiple_step_penalty
                self.stats.reward_components['step_penalties'] += 1
            else:
                reward += self.config.tutor_single_step_bonus
                self.stats.reward_components['step_bonuses'] += 1
                
            # Try completing from original solution up to wrong step
            partial_solution = "".join(solution_steps[:step_num])
            try:
                # Try completing with tutor's substitution
                completion_with_sub = await self.completion_agent.generate(
                    problem, 
                    partial_solution + substitution
                )
                
                # Try completing with original step
                completion_original = await self.completion_agent.generate(
                    problem,
                    partial_solution + solution_steps[step_num]
                )
                
                # Extract and compare answers
                sub_answer = extract_answer_from_solution(completion_with_sub)
                orig_answer = extract_answer_from_solution(completion_original)
                
                if sub_answer and orig_answer:
                    sub_numeric, _ = extract_numeric_answer(sub_answer)
                    orig_numeric, _ = extract_numeric_answer(orig_answer)
                    correct_numeric, _ = extract_numeric_answer(correct_answer)
                    
                    if all(x is not None for x in [sub_numeric, orig_numeric, correct_numeric]):
                        sub_correct = abs(sub_numeric - correct_numeric) <= self.config.numeric_tolerance
                        orig_correct = abs(orig_numeric - correct_numeric) <= self.config.numeric_tolerance
                        
                        if sub_correct and not orig_correct:
                            # Tutor's substitution leads to correct answer while original doesn't
                            reward = self.config.tutor_full_reward
                            self.stats.full_reward_reasons['step_correction'] += 1
                        elif orig_correct:
                            # Original step was actually correct
                            return 0.0
                            
            except Exception as e:
                self.logger.warning(f"Error during step validation: {str(e)}")
                return reward
                
        # Update base statistics
        self.stats.reward_components['base_rewards'] += 1
        if substitution:
            length_penalty = len(substitution) * self.config.tutor_substitution_length_cost
            reward += self.config.tutor_substitution_reward - length_penalty
            self.stats.reward_components['substitution_rewards'] += 1
            self.stats.reward_components['total_substitution_length_penalty'] += length_penalty
            
        return reward
        
    def calculate_reward(self, completion: str, **kwargs) -> float:
        """Synchronous wrapper for async reward calculation"""
        return asyncio.run(self.calculate_reward_async(completion, **kwargs))
