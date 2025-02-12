import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class RewardStats:
    """Base class for tracking reward statistics"""
    def __init__(self, config):
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
