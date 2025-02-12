import re
import json
import asyncio
import torch
import logging
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import os, sys
from typing import List, Optional, Tuple, Any, Union
from transformers import AutoTokenizer, AutoModel
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer, validate_solution, get_model
from utils.agents import CompletionAgent
from abc import ABC, abstractmethod
from config import RewardConfig
from utils.benchmark_config import BenchmarkConfig
from reward_stats import RewardStats

class BaseReward(ABC):
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
            'multiple_steps_in_substitution': 0,
            'polar_verdict_count': 0,
            'step_verdict_count': 0,
            'invalid_verdict_format': 0
        }

        # Track completion validation stats
        self.validation_stats = {
            'completion_attempts': 0,
            'successful_completions': 0,
            'failed_completions': 0,
            'completion_timeouts': 0,
            'completion_errors': 0
        }

        # Track step validation stats
        self.step_stats = {
            'step_identifications': 0,
            'valid_step_corrections': 0,
            'invalid_step_corrections': 0,
            'step_completion_rate': 0.0
        }

        # Track analysis quality metrics
        self.analysis_stats = {
            'analysis_length_distribution': {},
            'analysis_with_steps': 0,
            'analysis_without_steps': 0,
            'average_analysis_length': 0.0,
            'total_analysis_length': 0
        }
        
        # Track reward components
        self.reward_components = {
            'base_rewards': 0,
            'validation_rewards': 0,
            'analysis_rewards': 0,
            'substitution_rewards': 0,
            'step_bonuses': 0,
            'step_penalties': 0,
            'total_analysis_length_penalty': 0.0,
            'total_substitution_length_penalty': 0.0,
            'redundant_substitution_penalties': 0,
            'wrong_boxed_answer_penalties': 0,
            'majority_bonuses': 0,
            'diversity_bonuses': 0,
            'improvement_bonuses': {
                '0.1': 0,  # 10-40% completions
                '0.2': 0,  # 40-70% completions
                '0.3': 0,  # >70% completions
                'total': 0  # Total count of improvement bonuses
            }
        }
        
        # Track group-specific stats
        self.group_stats = {
            'unique_solutions': 0,
            'similar_solutions': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'total_similarity': 0.0,
            'majority_bonuses': 0,
            'diversity_bonuses': 0,
            # Voting statistics
            'total_votes': 0,
            'majority_votes': 0,
            'minority_votes': 0,
            'unanimous_correct': 0,
            'unanimous_incorrect': 0,
            'split_votes': 0,
            'majority_size_dist': {},  # Distribution of majority sizes
            'vote_margins': [],  # List of vote margins (majority - minority)
            'average_majority_size': 0.0,
            'average_vote_margin': 0.0
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
            
    def save_statistics(self, output_dir: str) -> None:
        """Save current statistics to JSON
        
        Args:
            output_dir: Directory to save statistics files
            
        The statistics are saved with timestamps and include:
        - Total number of batches processed
        - Total rewards accumulated
        - Distribution of rewards
        - Training duration
        - Component-specific statistics
        """
        try:
            stats_dir = Path(output_dir) / self.config.stats_dir
            stats_dir.mkdir(exist_ok=True)
            
            stats = {
                'total_batches': self.total_batches,
                'total_rewards': self.total_rewards,
                'reward_distribution': {str(k): v for k, v in self.reward_distribution.items()},
                'training_duration': str(datetime.now() - self.start_time),
                'section_stats': self.section_stats,
                'reward_components': self.reward_components,
                'group_stats': self.group_stats,
                'full_reward_reasons': self.full_reward_reasons
            }
            
            stats_file = stats_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save statistics: {str(e)}")
            
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
            f"\nVerdict Statistics:\n"
            f"  Polar verdicts: {self.section_stats['polar_verdict_count']}\n"
            f"  Step verdicts: {self.section_stats['step_verdict_count']}\n"
            f"  Invalid formats: {self.section_stats['invalid_verdict_format']}\n"
            f"\nSection Issues:\n"
            f"  Missing analysis: {self.section_stats['missing_analysis']}\n"
            f"  Missing verdict: {self.section_stats['missing_verdict']}\n"
            f"  Step verdict without substitution: {self.section_stats['step_verdict_without_substitution']}\n"
            f"  Polar verdict with substitution: {self.section_stats['polar_verdict_with_substitution']}\n"
            f"  Multiple steps in substitution: {self.section_stats['multiple_steps_in_substitution']}\n"
            f"\nCompletion Validation:\n"
            f"  Total attempts: {self.validation_stats['completion_attempts']}\n"
            f"  Successful: {self.validation_stats['successful_completions']}\n"
            f"  Failed: {self.validation_stats['failed_completions']}\n"
            f"  Timeouts: {self.validation_stats['completion_timeouts']}\n"
            f"  Errors: {self.validation_stats['completion_errors']}\n"
            f"\nStep Validation:\n"
            f"  Total identifications: {self.step_stats['step_identifications']}\n"
            f"  Valid corrections: {self.step_stats['valid_step_corrections']}\n"
            f"  Invalid corrections: {self.step_stats['invalid_step_corrections']}\n"
            f"  Completion rate: {self.step_stats['step_completion_rate']:.2%}\n"
            f"\nAnalysis Quality:\n"
            f"  With steps: {self.analysis_stats['analysis_with_steps']}\n"
            f"  Without steps: {self.analysis_stats['analysis_without_steps']}\n"
            f"  Average length: {self.analysis_stats['average_analysis_length']:.1f}\n"
            f"\nReward Components:\n"
            f"  Base rewards: {self.reward_components['base_rewards']}\n"
            f"  Validation rewards: {self.reward_components.get('validation_rewards', 0)}\n"
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
            f"  Majority bonuses: {self.reward_components.get('majority_bonuses', 0)}\n"
            f"  Diversity bonuses: {self.reward_components.get('diversity_bonuses', 0)}\n"
            f"  Unique solutions: {self.group_stats.get('unique_solutions', 0)}\n"
            f"  Similar solutions: {self.group_stats.get('similar_solutions', 0)}\n"
            f"  Average similarity: {self.group_stats.get('total_similarity', 0)/total_samples if total_samples else 0:.3f}\n"
            f"\nFull Reward Reasons:\n"
            f"  Correct answer: {self.full_reward_reasons['correct_answer']}\n"
            f"  Wrong approach: {self.full_reward_reasons['wrong_approach']}\n"
            f"  Step correction: {self.full_reward_reasons['step_correction']}\n"
            f"  Final step correct: {self.full_reward_reasons['final_step_correct']}"
        )


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
        
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Synchronous batch processing that runs async code in event loop"""
        # Validate inputs
        prompts = kwargs.get('prompts', [])
        answers = kwargs.get('answer') or kwargs.get('correct_answer', [])
        
        if len(completions) != len(prompts) or len(completions) != len(answers):
            self.logger.error(f"Mismatched lengths: completions={len(completions)}, prompts={len(prompts)}, answers={len(answers)}")
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
            prompt_groups[prompt]['answers'].append(ans)
            prompt_groups[prompt]['indices'].append(idx)
            
        # Process completions in parallel using event loop
        async def process_batch():
            tasks = []
            
            # Extract problems and solutions from kwargs if present
            problems = kwargs.get('problem', [''] * len(prompts))
            solutions = kwargs.get('model_solution', [''] * len(prompts))
            
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
                        'answer': ans,
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
        
        # Update stats and print batch summary
        self.stats.update(rewards, completions=completions)
        
        # Print reward-specific statistics summary every batch
        self.logger.info("\nReward Statistics Summary:")
        self.logger.info(self.stats.get_summary(getattr(self, 'relevant_stats', None)))
        
        return rewards

class SolutionReward(BaseReward):
    """Reward class for basic solution evaluation"""
    
    __name__ = "solution_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'validation_rewards', 'total_length_penalty'],
        'group_stats': ['correct_answers', 'incorrect_answers', 'valid_solutions', 'invalid_solutions', 'total_length']
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a single completion"""
        try:
            # Extract and validate the answer
            model_answer = extract_answer_from_solution(completion)
            if model_answer is None:
                self.logger.debug("No boxed answer found in completion")
                return 0.0
                
            # Convert to numeric values
            model_numeric, _ = extract_numeric_answer(model_answer)
            
            # Get correct answer from kwargs
            correct_answer = kwargs.get('answer')
            if not correct_answer:
                self.logger.warning("No correct answer provided in kwargs")
                return 0.0
            
            # Handle different input formats
            if isinstance(correct_answer, (list, tuple)):
                if not correct_answer:
                    self.logger.warning("Empty correct answer list")
                    return 0.0
                correct_answer = correct_answer[0]
                self.logger.debug(f"Using first element from list: {correct_answer}")
            elif isinstance(correct_answer, dict):
                correct_answer = str(correct_answer.get('answer', ''))
                self.logger.debug(f"Extracted answer from dict: {correct_answer}")
            
            # Convert to string if needed
            correct_answer = str(correct_answer)
            correct_numeric, _ = extract_numeric_answer(correct_answer)
            
            if model_numeric is None or correct_numeric is None:
                self.logger.debug(f"Could not extract numeric values - Model: {model_numeric}, Correct: {correct_numeric}")
                return 0.0
                
            # Initialize reward
            reward = 0.0
            
            # Check correctness
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            self.logger.info(f"Correctness check - Model: {model_numeric:.6f}, Expected: {correct_numeric:.6f}, Correct: {is_correct}")
            
            if is_correct:
                reward = self.config.base_reward
                self.logger.info(f"Applied base reward: +{self.config.base_reward:.3f}")
                
            # Add validation reward
            is_valid, validation_msg = validate_solution(completion)
            self.logger.info(f"Validation check - Valid: {is_valid}, Message: {validation_msg}")
            if is_valid:
                reward += self.config.validation_reward
                self.logger.info(f"Applied validation reward: +{self.config.validation_reward:.3f}")
                
            # Apply length penalty
            length_penalty = len(completion) * self.config.length_penalty_factor
            reward -= length_penalty
            self.logger.info(f"Applied length penalty: -{length_penalty:.3f} (length: {len(completion)})")
            
            # Update detailed statistics
            if is_correct:
                self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + 1
                self.stats.group_stats['correct_answers'] = self.stats.group_stats.get('correct_answers', 0) + 1
            else:
                self.stats.group_stats['incorrect_answers'] = self.stats.group_stats.get('incorrect_answers', 0) + 1
                
            if is_valid:
                self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + 1
                self.stats.group_stats['valid_solutions'] = self.stats.group_stats.get('valid_solutions', 0) + 1
            else:
                self.stats.group_stats['invalid_solutions'] = self.stats.group_stats.get('invalid_solutions', 0) + 1
                
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            self.stats.group_stats['total_length'] = self.stats.group_stats.get('total_length', 0) + len(completion)
            self.stats.group_stats['total_solutions'] = self.stats.group_stats.get('total_solutions', 0) + 1
            
            # Print statistics summary every 100 batches
            if self.stats.total_batches % 100 == 0:
                self.logger.info("\nSolution Statistics:")
                self.logger.info(f"Total solutions: {self.stats.group_stats.get('total_solutions', 0)}")
                self.logger.info(f"Correct answers: {self.stats.group_stats.get('correct_answers', 0)}")
                self.logger.info(f"Incorrect answers: {self.stats.group_stats.get('incorrect_answers', 0)}")
                self.logger.info(f"Valid solutions: {self.stats.group_stats.get('valid_solutions', 0)}")
                self.logger.info(f"Invalid solutions: {self.stats.group_stats.get('invalid_solutions', 0)}")
                self.logger.info(f"Average length: {self.stats.group_stats.get('total_length', 0) / max(1, self.stats.group_stats.get('total_solutions', 1)):.1f}")
                self.logger.info(f"Total length penalty: {self.stats.reward_components.get('total_length_penalty', 0.0):.3f}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
            

class SolutionSimilarityChecker:
    """Handles embedding and similarity computation for solutions"""
    def __init__(self, config: RewardConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        self.model = AutoModel.from_pretrained(config.embedding_model)
        
        # Move model to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Ensure all parameters are frozen and on correct device
        with torch.no_grad():
            for param in self.model.parameters():
                param.requires_grad_(False)  # Use requires_grad_ method
                if param.device != self.device:
                    param.data = param.data.to(self.device)

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():  # Ensure no gradients are tracked
            with torch.amp.autocast('cuda', enabled=True):
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.embedding_max_length,
                    return_tensors="pt"
                )
                # Move inputs to device and ensure they don't track gradients
                inputs = {k: v.to(self.device).detach() for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                # Detach embeddings from computation graph
                embeddings = outputs.last_hidden_state.mean(dim=1).detach()
                # Normalize and ensure no gradients
                normalized = F.normalize(embeddings, p=2, dim=1)
                return normalized.detach()

    def compute_similarity_matrix(self, solutions: List[str]) -> torch.Tensor:
        with torch.no_grad():  # Ensure no gradients are tracked
            embeddings = self.get_embeddings(solutions)
            similarity_matrix = torch.mm(embeddings, embeddings.t())
            return similarity_matrix.detach()  # Ensure result is detached

class GroupReward(BaseReward):
    """Reward class for group-based solution evaluation"""
    
    __name__ = "group_reward"
    relevant_stats = {
        'reward_components': ['base_rewards', 'majority_bonuses', 'diversity_bonuses'],
        'group_stats': [
            'correct_answers', 'incorrect_answers', 'unique_solutions', 'similar_solutions',
            'total_similarity', 'majority_votes', 'minority_votes', 'unanimous_correct',
            'unanimous_incorrect', 'split_votes', 'majority_size_dist', 'vote_margins',
            'average_majority_size', 'average_vote_margin'
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
            print(len(group_completions),group_idx)
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
            correct_numeric, _ = extract_numeric_answer(kwargs['answer'])
            
            if model_numeric is None or correct_numeric is None:
                self.logger.debug("Could not extract numeric values - returning 0.0")
                return 0.0
                
            # Calculate base reward
            is_correct = abs(model_numeric - correct_numeric) <= self.config.numeric_tolerance
            reward = self.config.group_base_reward if is_correct else 0.0
            if is_correct:
                self.stats.reward_components['base_rewards'] += 1
            self.logger.info(f"Base calculation - Answer: {model_numeric:.6f}, Expected: {correct_numeric:.6f}, Correct: {is_correct}")
            
            # Add validation reward
            is_valid, validation_msg = validate_solution(completion)
            self.logger.info(f"Validation check - Valid: {is_valid}, Message: {validation_msg}")
            if is_valid:
                reward += self.config.validation_reward
                self.stats.reward_components['validation_rewards'] = self.stats.reward_components.get('validation_rewards', 0) + 1
                self.logger.info(f"Applied validation reward: +{self.config.validation_reward:.3f}")
            
            # Calculate similarity matrix for group
            similarity_matrix = self.similarity_checker.compute_similarity_matrix(group_completions)
            
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
            
            # Majority bonus - only apply if group has more than one completion
            if len(group_completions) > 1:
                correct_count = sum(all_results)
                incorrect_count = len(group_completions) - correct_count
                majority_size = max(correct_count, incorrect_count)
                minority_size = min(correct_count, incorrect_count)
                vote_margin = majority_size - minority_size
                
                # Update voting statistics
                self.stats.group_stats['total_votes'] += 1
                if majority_size == len(group_completions):
                    if correct_count > incorrect_count:
                        self.stats.group_stats['unanimous_correct'] += 1
                    else:
                        self.stats.group_stats['unanimous_incorrect'] += 1
                else:
                    self.stats.group_stats['split_votes'] += 1
                
                # Track majority size distribution
                self.stats.group_stats['majority_size_dist'][majority_size] = \
                    self.stats.group_stats['majority_size_dist'].get(majority_size, 0) + 1
                
                # Update vote margins
                self.stats.group_stats['vote_margins'].append(vote_margin)
                total_votes = self.stats.group_stats['total_votes']
                self.stats.group_stats['average_majority_size'] = \
                    (majority_size + (total_votes - 1) * self.stats.group_stats['average_majority_size']) / total_votes
                self.stats.group_stats['average_vote_margin'] = \
                    (vote_margin + (total_votes - 1) * self.stats.group_stats['average_vote_margin']) / total_votes
                
                # Determine if current completion is in majority
                is_in_majority = (is_correct and correct_count > incorrect_count) or \
                                (not is_correct and incorrect_count > correct_count)
                majority_bonus = self.config.group_majority_bonus if is_correct else self.config.group_majority_bonus * 0.1
                
                self.logger.info(f"Majority calculation - Correct count: {correct_count}/{len(group_completions)}, "
                               f"In majority: {is_in_majority}, Margin: {vote_margin}")
                
                if is_in_majority:
                    reward += majority_bonus
                    self.stats.reward_components['majority_bonuses'] = self.stats.reward_components.get('majority_bonuses', 0) + 1
                    self.stats.group_stats['majority_votes'] += 1
                    self.logger.info(f"Applied majority bonus: +{majority_bonus:.3f}")
                else:
                    self.stats.group_stats['minority_votes'] += 1
                    
                # Diversity bonus
                similarities = similarity_matrix[group_idx]
                similarities[group_idx] = 0  # Remove self-similarity
                avg_similarity = similarities.mean().item()
                
                self.logger.info(f"Similarity calculation - Average similarity: {avg_similarity:.3f}")
                
                diversity_bonus = 0
                if avg_similarity < self.config.group_similarity_threshold_low:  # Unique solution
                    diversity_bonus = self.config.group_diversity_bonus if is_correct else self.config.group_diversity_bonus * 0.1
                    reward += diversity_bonus
                    self.stats.reward_components['diversity_bonuses'] = self.stats.reward_components.get('diversity_bonuses', 0) + 1
                    self.logger.info(f"Applied uniqueness bonus: +{diversity_bonus:.3f}")
                elif avg_similarity > self.config.group_similarity_threshold_high:  # Very similar to others
                    diversity_bonus = -(self.config.group_diversity_bonus / 2 if is_correct else self.config.group_diversity_bonus * 0.05)
                    reward += diversity_bonus
                    self.logger.info(f"Applied similarity penalty: {diversity_bonus:.3f}")
                
            # Update group-specific statistics
            if is_correct:
                self.stats.group_stats['correct_answers'] += 1
            else:
                self.stats.group_stats['incorrect_answers'] += 1
                
            if is_in_majority:
                self.stats.group_stats['majority_bonuses'] = self.stats.group_stats.get('majority_bonuses', 0) + 1
            if diversity_bonus > 0:
                self.stats.group_stats['diversity_bonuses'] = self.stats.group_stats.get('diversity_bonuses', 0) + 1
                
            if avg_similarity < self.config.group_similarity_threshold_low:
                self.stats.group_stats['unique_solutions'] += 1
            elif avg_similarity > self.config.group_similarity_threshold_high:
                self.stats.group_stats['similar_solutions'] += 1
                
            self.stats.group_stats['total_similarity'] += avg_similarity
            
            # Track length penalties
            length_penalty = len(completion) * self.config.length_penalty_factor
            self.stats.group_stats['total_length_penalty'] = self.stats.group_stats.get('total_length_penalty', 0.0) + length_penalty
            self.stats.group_stats['total_length'] = self.stats.group_stats.get('total_length', 0) + len(completion)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating group reward: {str(e)}")
            return 0.0

class TutorReward(BaseReward):
    """Reward class for tutor response evaluation"""
    
    __name__ = "tutor_reward"
    relevant_stats = {
        'section_stats': [
            'missing_analysis', 'missing_verdict', 'missing_substitution',
            'invalid_step_number', 'polar_verdict_with_substitution',
            'step_verdict_without_substitution', 'multiple_steps_in_substitution',
            'polar_verdict_count', 'step_verdict_count', 'invalid_verdict_format'
        ],
        'validation_stats': ['completion_attempts', 'successful_completions', 'failed_completions'],
        'step_stats': ['step_identifications', 'valid_step_corrections', 'invalid_step_corrections', 'step_completion_rate'],
        'analysis_stats': ['analysis_with_steps', 'analysis_without_steps', 'average_analysis_length'],
        'reward_components': [
            'base_rewards', 'analysis_rewards', 'substitution_rewards', 'step_bonuses',
            'step_penalties', 'total_analysis_length_penalty', 'total_substitution_length_penalty',
            'redundant_substitution_penalties', 'wrong_boxed_answer_penalties'
        ],
        'full_reward_reasons': ['correct_answer', 'wrong_approach', 'step_correction', 'final_step_correct']
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        # Load benchmark config and override with reward config values
        benchmark_config = BenchmarkConfig.from_args('Benchmark config for reward calculation')
        benchmark_config.auxiliary_port = config.completion_port
        benchmark_config.auxiliary_temp = config.completion_temp
        
        # Create auxiliary model with temperature
        auxiliary = get_model(benchmark_config, role="auxiliary")
        # Initialize completion agent for validation
        self.completion_agent = CompletionAgent(auxiliary)
        
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
        
    async def _validate_completions(
        self,
        problem: str,
        partial_solution: str,
        correct_answer: str,
        num_attempts: int = 10
    ) -> Tuple[int, int]:
        """Try completions in parallel until finding a successful one."""
        async def try_completion():
            try:
                completion = await self.completion_agent.generate(problem, partial_solution)
                complete_solution = partial_solution + completion
                
                model_answer = extract_answer_from_solution(complete_solution)
                if model_answer is None:
                    self.logger.debug("No boxed answer found in completion")
                    return False
                    
                numeric_answer, debug_info = extract_numeric_answer(model_answer, debug=True)
                correct_numeric, _ = extract_numeric_answer(correct_answer)
                
                if numeric_answer is None:
                    self.logger.debug(f"Could not extract numeric answer: {debug_info}")
                    return False
                    
                if correct_numeric is None:
                    self.logger.debug(f"Could not extract correct numeric answer from: {correct_answer}")
                    return False
                    
                is_correct = abs(numeric_answer - correct_numeric) <= self.config.numeric_tolerance
                
                return is_correct
                
            except Exception as e:   
                self.logger.warning(f"Completion attempt failed: {str(e)}")
                return False
    
        # Track completion attempts
        self.stats.validation_stats['completion_attempts'] += num_attempts
        
        try:
            # Run all completion attempts in parallel
            results = await asyncio.gather(*[try_completion() for _ in range(num_attempts)])
            successful = sum(1 for r in results if r)
            
            # Update completion stats
            self.stats.validation_stats['successful_completions'] += successful
            self.stats.validation_stats['failed_completions'] += (len(results) - successful)
            
            return successful, len(results)
            
        except asyncio.TimeoutError:
            self.stats.validation_stats['completion_timeouts'] += 1
            return 0, num_attempts
        except Exception as e:
            self.stats.validation_stats['completion_errors'] += 1
            return 0, num_attempts

    async def _validate_whole_approach_is_wrong(
        self,
        problem: str,
        solution: str,
        correct_answer: str
    ) -> bool:
        """Validate that the analysis section alone can lead to correct completions"""
        # Split solution into steps and get the analysis part
        steps = self.split_into_steps(solution)
        if not steps:
            return False
            
        # First part before steps is the analysis
        analysis = steps[0]
        
        # Try completions starting with just the analysis
        successful, total = await self._validate_completions(
            problem,
            analysis,
            correct_answer,
            self.config.completion_attempts
        )
        
        return successful == 0 and total == self.config.completion_attempts

    async def _validate_step_identification(
        self,
        problem: str,
        steps: List[str],
        step_num: int,
        substitution: str,
        correct_answer: str,
        original_step: str
    ) -> Tuple[bool, float]:
        """Validate step identification and correction in parallel.
        Returns (is_valid, improvement_bonus)"""
        
        # Run both validations in parallel
        wrong_partial = "".join(steps[:step_num+1])  # Include the wrong step
        corrected_partial = "".join(steps[:step_num]) + substitution  # Replace wrong step
        
        wrong_check, fixed_check = await asyncio.gather(
            self._validate_completions(problem, wrong_partial, correct_answer, self.config.completion_attempts),
            self._validate_completions(problem, corrected_partial, correct_answer, self.config.completion_attempts)
        )

        successful_wrong, total_wrong = wrong_check
        successful_fixed, total_fixed = fixed_check

        # Calculate improvement bonus based on relative success rate
        improvement_bonus = 0.0
        if successful_wrong == 0:  # Only reward if original step had no successful completions
            success_rate = successful_fixed / total_fixed
            if 0.1 < success_rate <= 0.4:  # 10-40%
                improvement_bonus = 0.1
            elif 0.4 < success_rate <= 0.7:  # 40-70%
                improvement_bonus = 0.2
            elif success_rate > 0.7:         # >70%
                improvement_bonus = 0.3
        
        is_valid = successful_wrong == 0 and successful_fixed > 0
        return is_valid, improvement_bonus

    async def calculate_reward(self, tutor_response: str, **kwargs) -> float:
        """Calculate reward for a tutor's evaluation of a solution"""
        # First verify if model solution is correct
        problem = kwargs.get('problem')
        student_solution = kwargs.get('solution')
        correct_answer = kwargs.get('answer')
        student_answer = extract_answer_from_solution(student_solution)
        if student_answer is None:
            self.logger.warning(f"No boxed answer found in model solution: {student_solution[:100]}...")
            return 0.0
        student_numeric, _ = extract_numeric_answer(student_answer)
        correct_numeric, _ = extract_numeric_answer(str(correct_answer))
        if student_numeric is None or correct_numeric is None:
            self.logger.warning(f"Could not extract numeric values - Model: {student_answer}, Correct: {kwargs.get('correct_answer', '')}")
            return 0.0

        # Extract sections from tutor's response
        analysis, verdict, substitution = self.extract_sections(tutor_response)
        
        if verdict is None:
            self.logger.debug(f"Missing verdict section in tutor response: {tutor_response[:100]}...")
            self.stats.section_stats['invalid_verdict_format'] += 1
            return 0.0

        # Track analysis stats
        if analysis:
            length = len(analysis)
            self.stats.analysis_stats['total_analysis_length'] += length
            self.stats.analysis_stats['analysis_length_distribution'][length] = \
                self.stats.analysis_stats['analysis_length_distribution'].get(length, 0) + 1
            
            # Update average length
            total_analyses = sum(self.stats.analysis_stats['analysis_length_distribution'].values())
            self.stats.analysis_stats['average_analysis_length'] = \
                self.stats.analysis_stats['total_analysis_length'] / total_analyses

            # Check if analysis contains step references
            if any(f"step {i}" in analysis.lower() for i in range(1, 10)):
                self.stats.analysis_stats['analysis_with_steps'] += 1
            else:
                self.stats.analysis_stats['analysis_without_steps'] += 1
        
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
                
            # Track step validation
            self.stats.step_stats['step_identifications'] += 1
            self.logger.info(f"Validating step {step_num} correction")
        
            # Check if substitution has multiple steps
            substitution_steps = self.split_into_steps(substitution)
            if len(substitution_steps) > 1:
                reward -= self.config.tutor_multiple_step_penalty
                self.stats.reward_components['step_penalties'] += 1
                self.stats.step_stats['invalid_step_corrections'] += 1
            else:
                reward += self.config.tutor_single_step_bonus
                self.stats.reward_components['step_bonuses'] += 1
                self.stats.step_stats['valid_step_corrections'] += 1
            
            # Update completion rate
            total_corrections = self.stats.step_stats['valid_step_corrections'] + \
                              self.stats.step_stats['invalid_step_corrections']
            if total_corrections > 0:
                self.stats.step_stats['step_completion_rate'] = \
                    self.stats.step_stats['valid_step_corrections'] / total_corrections
                
            # Try completing from original solution up to wrong step
            partial_solution = "".join(solution_steps[:step_num])
            try:
                # Try multiple completions for both original and substituted steps
                wrong_partial = partial_solution + solution_steps[step_num]
                corrected_partial = partial_solution + substitution
                
                wrong_check, fixed_check = await asyncio.gather(
                    self._validate_completions(
                        problem, 
                        wrong_partial, 
                        correct_answer,
                        self.config.completion_attempts
                    ),
                    self._validate_completions(
                        problem,
                        corrected_partial,
                        correct_answer,
                        self.config.completion_attempts
                    )
                )
                
                successful_wrong, total_wrong = wrong_check
                successful_fixed, total_fixed = fixed_check
                
                self.logger.info(f"Completion results - Original: {successful_wrong}/{total_wrong}, Fixed: {successful_fixed}/{total_fixed}")
                
                if successful_wrong == 0 and successful_fixed > 0:
                    # Calculate improvement bonus based on success rate
                    success_rate = successful_fixed / total_fixed
                    improvement_bonus = 0.0
                    
                    if 0.1 < success_rate <= 0.4:  # 10-40%
                        improvement_bonus = 0.1
                    elif 0.4 < success_rate <= 0.7:  # 40-70%
                        improvement_bonus = 0.2
                    elif success_rate > 0.7:  # >70%
                        improvement_bonus = 0.3
                        
                    reward = self.config.tutor_full_reward + improvement_bonus
                    self.stats.full_reward_reasons['step_correction'] += 1
                    self.logger.info(f"Step correction successful - Success rate: {success_rate:.2%}, Bonus: {improvement_bonus}")
                    
                    # Track improvement bonus
                    if improvement_bonus > 0:
                        bonus_key = str(improvement_bonus)
                        self.stats.reward_components['improvement_bonuses'][bonus_key] = \
                            self.stats.reward_components['improvement_bonuses'].get(bonus_key, 0) + 1
                        self.stats.reward_components['improvement_bonuses']['total'] += 1
                        
                elif successful_wrong > 0:
                    # Original step was actually correct
                    self.logger.warning("Original step was actually correct - returning 0 reward")
                    return 0.0
                            
            except Exception as e:
                self.logger.warning(f"Error during step validation: {str(e)}")
                return reward
                
        # Update base statistics
        if reward >= self.config.tutor_structure_base_reward:
            self.stats.reward_components['base_rewards'] = self.stats.reward_components.get('base_rewards', 0) + 1
            
        # Track section-level stats
        if not analysis:
            self.stats.section_stats['missing_analysis'] += 1
        if not verdict:
            self.stats.section_stats['missing_verdict'] += 1
        if not substitution and verdict.startswith("Step "):
            self.stats.section_stats['missing_substitution'] += 1
            
        # Track substitution stats
        if substitution:
            length_penalty = len(substitution) * self.config.tutor_substitution_length_cost
            reward += self.config.tutor_substitution_reward - length_penalty
            self.stats.reward_components['substitution_rewards'] = self.stats.reward_components.get('substitution_rewards', 0) + 1
            self.stats.reward_components['total_substitution_length_penalty'] = self.stats.reward_components.get('total_substitution_length_penalty', 0.0) + length_penalty
            
            if verdict in polar_verdicts:
                self.stats.section_stats['polar_verdict_with_substitution'] += 1
                
            substitution_steps = self.split_into_steps(substitution)
            if len(substitution_steps) > 1:
                self.stats.section_stats['multiple_steps_in_substitution'] += 1
                
        # Track improvement bonuses if applicable
        if verdict.startswith("Step ") and reward == self.config.tutor_full_reward:
            bonus_level = None
            if reward > self.config.tutor_full_reward:
                bonus = reward - self.config.tutor_full_reward
                if abs(bonus - 0.1) < 1e-6:
                    bonus_level = '0.1'
                elif abs(bonus - 0.2) < 1e-6:
                    bonus_level = '0.2'
                elif abs(bonus - 0.3) < 1e-6:
                    bonus_level = '0.3'
                    
            if bonus_level:
                self.stats.reward_components['improvement_bonuses'][bonus_level] = \
                    self.stats.reward_components['improvement_bonuses'].get(bonus_level, 0) + 1
                self.stats.reward_components['improvement_bonuses']['total'] += 1
                
        return reward
        
