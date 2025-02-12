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
            'average_vote_margin': 0.0,
            # Length and content penalties
            'total_length_penalty': 0.0,
            'total_length': 0,
            'total_analysis_length_penalty': 0.0,
            'total_substitution_length_penalty': 0.0
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
            
    def get_summary(self, relevant_stats=None) -> str:
        """Get a human-readable summary of statistics
        
        Args:
            relevant_stats: Dictionary mapping stat categories to lists of relevant stat names
        """
        total_samples = sum(self.reward_distribution.values())
        if total_samples == 0:
            return "No samples processed yet"
            
        elapsed = datetime.now() - self.start_time
        avg_reward = self.total_rewards / total_samples if total_samples > 0 else 0
        
        # Dynamic binning for rewards
        rewards = sorted(self.reward_distribution.keys())
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            # Create 10 bins or less if we have fewer unique rewards
            num_bins = min(10, len(rewards))
            if num_bins > 1:
                bin_size = (max_reward - min_reward) / num_bins
                bins = {}
                for reward, count in self.reward_distribution.items():
                    if bin_size == 0:  # Handle case where all rewards are the same
                        bin_idx = 0
                    else:
                        bin_idx = min(int((reward - min_reward) / bin_size), num_bins - 1)
                    bin_start = min_reward + bin_idx * bin_size
                    bin_end = min_reward + (bin_idx + 1) * bin_size
                    bin_key = f"{bin_start:.3f} to {bin_end:.3f}"
                    bins[bin_key] = bins.get(bin_key, 0) + count
                
                # Format reward distribution with bins
                reward_dist_str = "\n".join(
                    f"  {bin_range}: {count} samples ({count/total_samples*100:.1f}%)" 
                    for bin_range, count in sorted(bins.items())
                )
            else:
                # If only one unique reward, show it directly
                reward_dist_str = f"  {rewards[0]:.6f}: {total_samples} samples (100%)"
        else:
            reward_dist_str = "No rewards recorded"

        # Always show basic stats
        summary = [
            f"Training time: {elapsed}",
            f"Processed {self.total_batches} batches",
            f"Average reward: {avg_reward:.6f}",
            f"Total samples: {total_samples}",
            f"\nReward Distribution:\n{reward_dist_str}"
        ]
        
        if not relevant_stats:
            # If no relevant stats specified, show everything
            relevant_stats = {
                'section_stats': list(self.section_stats.keys()),
                'validation_stats': list(self.validation_stats.keys()),
                'step_stats': list(self.step_stats.keys()),
                'analysis_stats': list(self.analysis_stats.keys()),
                'reward_components': list(self.reward_components.keys()),
                'group_stats': list(self.group_stats.keys()),
                'full_reward_reasons': list(self.full_reward_reasons.keys())
            }
        
        # Build sections based on relevant stats
        for category, stat_names in relevant_stats.items():
            if not stat_names:
                continue
                
            stats_dict = getattr(self, category)
            if not stats_dict:
                continue
                
            # Add section header
            section_name = category.replace('_', ' ').title()
            summary.append(f"\n{section_name}:")
            
            # Add relevant stats
            for stat_name in stat_names:
                if stat_name not in stats_dict:
                    continue
                    
                value = stats_dict[stat_name]
                # Format special cases
                if isinstance(value, float):
                    if 'penalty' in stat_name or 'reward' in stat_name:
                        formatted_value = f"{value:.6f}"
                    else:
                        formatted_value = f"{value:.3f}"
                elif isinstance(value, dict):
                    if stat_name == 'improvement_bonuses':
                        # Special formatting for improvement bonuses
                        bonus_counts = []
                        for bonus, count in value.items():
                            if bonus != 'total':
                                bonus_counts.append(f"+{bonus} bonus: {count}x")
                        formatted_value = f"Total: {value.get('total', 0)}\n    " + "\n    ".join(bonus_counts)
                    else:
                        formatted_value = str(value)
                elif isinstance(value, list):
                    if value:
                        formatted_value = f"avg: {sum(value)/len(value):.3f}, count: {len(value)}"
                    else:
                        formatted_value = "empty"
                else:
                    formatted_value = str(value)
                
                # Format stat name for display
                display_name = stat_name.replace('_', ' ').title()
                summary.append(f"  {display_name}: {formatted_value}")
        
        return "\n".join(summary)
