import re
import asyncio
import logging
import torch
import numpy as np # For dtw-python compatibility if needed
from dtw import dtw # dtw-python library
from typing import List, Dict, Tuple, Optional, Any
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from grpo.rewards.base_reward import BaseReward
from grpo.config import RewardConfig
from grpo.reward_stats import RewardStats
from utils.similarity_checker import SolutionSimilarityChecker
from utils.solution_utils import extract_answer_from_solution, extract_numeric_answer

class SolutionDTWReward(BaseReward):
    """
    Reward class that incorporates DTW similarity between sequences of solution steps,
    in addition to basic answer correctness.
    """
    __name__ = "solution_dtw_reward"
    relevant_stats = {
        'reward_components': ['base_correctness_rewards', 'dtw_similarity_rewards', 'step_count_match_rewards', 'total_rewards'],
        'dtw_stats': ['average_dtw_distance', 'completions_with_steps'],
        'step_count_stats': ['average_step_diff', 'perfect_step_count_matches'],
        'group_stats': [ # From SolutionReward, if we want to keep some basic answer stats
            'correct_answers', 'incorrect_answers', 'correct_to_total_ratio'
        ],
        'plurality_stats': [ # From SolutionReward
            'plurality_correct_rate', 'avg_plurality_percentage', 'avg_completion_length'
        ]
    }

    def __init__(self, config: RewardConfig, similarity_checker: SolutionSimilarityChecker):
        super().__init__(config)
        self.similarity_checker = similarity_checker
        self.step_pattern = re.compile(r"<step>(.*?)</step>", re.DOTALL) # Use < and > for XML safety

        # DTW specific config (can be added to RewardConfig later)
        self.dtw_max_reward = getattr(config, 'dtw_max_reward', 3.0) # Max reward for perfect DTW match (increased from 1.0)
        self.correctness_reward_value = getattr(config, 'correctness_reward', 1.0) # Reward for final answer correctness

        # Initialize stats
        if not hasattr(self.stats, 'dtw_stats'):
            self.stats.dtw_stats = {
                'total_dtw_distance': 0.0,
                'dtw_comparisons_count': 0,
                'average_dtw_distance': 0.0,
                'completions_with_steps': 0
            }
        # Ensure reward_components dictionary from RewardStats is initialized with our keys
        self.stats.reward_components.setdefault('base_correctness_rewards', 0.0)
        self.stats.reward_components.setdefault('dtw_similarity_rewards', 0.0)
        self.stats.reward_components.setdefault('step_count_match_rewards', 0.0)
        # total_rewards is usually handled by RewardStats.update or BaseReward

        self.step_count_match_max_reward = getattr(config, 'step_count_match_max_reward', 0.25) # Max reward for matching step counts
        if not hasattr(self.stats, 'step_count_stats'):
            self.stats.step_count_stats = {
                'total_step_diff': 0,
                'num_step_comparisons': 0,
                'average_step_diff': 0.0,
                'perfect_step_count_matches': 0
            }
        
        # For plurality and basic answer stats (mimicking SolutionReward)
        self.answer_grouping_tolerance = getattr(config, 'answer_grouping_tolerance', 1e-2)
        if not hasattr(self.stats, 'group_stats'):
            self.stats.group_stats = {
                'correct_answers': 0,
                'incorrect_answers': 0,
            }


    def _extract_steps(self, text: str) -> List[str]:
        # The user's prompt implies tags are <step> not <step>
        # Adjusting regex if tags are literal XML tags in the string
        # If the strings truly contain "<step>", the original regex was correct.
        # Assuming literal tags for now based on "completions have <step> tags"
        literal_step_pattern = re.compile(r"<step>(.*?)</step>", re.DOTALL)
        return literal_step_pattern.findall(text)

    async def _calculate_dtw_reward_component(
        self,
        completion_steps: List[str],
        reference_steps: List[str]
    ) -> float:
        self.logger.debug(f"DTW: Received {len(completion_steps)} completion steps and {len(reference_steps)} reference steps.")
        if len(completion_steps) > 0:
            self.logger.debug(f"DTW: First completion step (sample): '{completion_steps[0][:100]}...'")
        if len(reference_steps) > 0:
            self.logger.debug(f"DTW: First reference step (sample): '{reference_steps[0][:100]}...'")

        if not completion_steps or not reference_steps:
            self.logger.debug("DTW: One or both step lists are empty. Returning 0.0 DTW reward.")
            return 0.0

        try:
            # Get embeddings for all steps
            # Based on SolutionSimilarityChecker, get_embeddings is synchronous.
            
            comp_step_embeddings_tensor = self.similarity_checker.get_embeddings(completion_steps)
            ref_step_embeddings_tensor = self.similarity_checker.get_embeddings(reference_steps)

            self.logger.info(f"DTW: Comp step embeddings tensor shape: {comp_step_embeddings_tensor.shape}")
            self.logger.info(f"DTW: Ref step embeddings tensor shape: {ref_step_embeddings_tensor.shape}")

            if comp_step_embeddings_tensor.nelement() == 0 or ref_step_embeddings_tensor.nelement() == 0:
                self.logger.warning(f"DTW: Embeddings for steps resulted in zero elements. Comp shape: {comp_step_embeddings_tensor.shape}, Ref shape: {ref_step_embeddings_tensor.shape}. Returning 0.0 DTW reward.")
                self.logger.info(f"Problematic steps content: Completion Steps: {completion_steps}, Reference Steps: {reference_steps}")
                return 0.0

            embedding_dim = self.similarity_checker.model.config.hidden_size
            self.logger.info(f"DTW: Expected embedding dimension: {embedding_dim}")

            # Convert to NumPy arrays, ensuring they are 2D: (num_steps, embedding_dim)
            if comp_step_embeddings_tensor.nelement() > 0:
                np_comp_embeddings = comp_step_embeddings_tensor.cpu().numpy()
                if np_comp_embeddings.ndim == 1: # Should be (N,D), not (D,) or (N,)
                    self.logger.error(f"DTW: Comp step embeddings became 1D unexpectedly: {np_comp_embeddings.shape} from tensor shape {comp_step_embeddings_tensor.shape}")
                    self.logger.info(f"Problematic steps content: Completion Steps: {completion_steps}")
                    return 0.0 # Indicates an issue with get_embeddings or logic
            else: 
                np_comp_embeddings = np.empty((0, embedding_dim))
            
            if ref_step_embeddings_tensor.nelement() > 0:
                np_ref_embeddings = ref_step_embeddings_tensor.cpu().numpy()
                if np_ref_embeddings.ndim == 1:
                    self.logger.error(f"DTW: Ref step embeddings became 1D unexpectedly: {np_ref_embeddings.shape} from tensor shape {ref_step_embeddings_tensor.shape}")
                    self.logger.info(f"Problematic steps content: Reference Steps: {reference_steps}")
                    return 0.0
            else: 
                np_ref_embeddings = np.empty((0, embedding_dim))

            self.logger.info(f"DTW: Shape of np_comp_embeddings for DTW: {np_comp_embeddings.shape}")
            self.logger.info(f"DTW: Shape of np_ref_embeddings for DTW: {np_ref_embeddings.shape}")

            # --- Direct Printing of Full Embedding Values ---
            print("\n--- [RAW EMBEDDING VALUES] Completion Step Embeddings ---")
            for i, step_text in enumerate(completion_steps):
                if i < np_comp_embeddings.shape[0]:
                    step_emb_np = np_comp_embeddings[i]
                    clean_step_text = step_text[:70].replace('\n', ' ') # Corrected method
                    print(f"  Comp Step {i+1}/{len(completion_steps)} (Text: '{clean_step_text}...'):")
                    print(f"    Embedding (shape {step_emb_np.shape}): {step_emb_np.tolist()}")
                else:
                    clean_step_text = step_text[:70].replace('\n', ' ') # Corrected method
                    print(f"  Comp Step {i+1}/{len(completion_steps)} (Text: '{clean_step_text}...'): ERROR - No corresponding embedding in array (array shape {np_comp_embeddings.shape})")

            print("\n--- [RAW EMBEDDING VALUES] Reference Step Embeddings ---")
            for i, step_text in enumerate(reference_steps):
                if i < np_ref_embeddings.shape[0]:
                    step_emb_np = np_ref_embeddings[i]
                    clean_step_text = step_text[:70].replace('\n', ' ') # Corrected method
                    print(f"  Ref Step {i+1}/{len(reference_steps)} (Text: '{clean_step_text}...'):")
                    print(f"    Embedding (shape {step_emb_np.shape}): {step_emb_np.tolist()}")
                else:
                    clean_step_text = step_text[:70].replace('\n', ' ') # Corrected method
                    print(f"  Ref Step {i+1}/{len(reference_steps)} (Text: '{clean_step_text}...'): ERROR - No corresponding embedding in array (array shape {np_ref_embeddings.shape})")
            print("--- [RAW EMBEDDING VALUES] End of Prints ---\n")
            # --- End of Direct Printing ---

            # Ensure both are 2D. This is a safeguard; previous checks should handle most issues.
            if np_comp_embeddings.ndim != 2 or np_ref_embeddings.ndim != 2:
                 self.logger.error(f"DTW: Converted step embeddings are not 2D. Comp: {np_comp_embeddings.shape}, Ref: {np_ref_embeddings.shape}. Returning 0.0 DTW reward.")
                 self.logger.info(f"Problematic steps content: Completion Steps: {completion_steps}, Reference Steps: {reference_steps}")
                 return 0.0
            
            comp_cols = np_comp_embeddings.shape[1] if np_comp_embeddings.shape[0] > 0 else embedding_dim
            ref_cols = np_ref_embeddings.shape[1] if np_ref_embeddings.shape[0] > 0 else embedding_dim

            if comp_cols != ref_cols:
                self.logger.error(f"DTW: Embedding dimensions (columns) mismatch! Comp_cols: {comp_cols} (from shape {np_comp_embeddings.shape}), Ref_cols: {ref_cols} (from shape {np_ref_embeddings.shape}). Returning 0.0 DTW reward.")
                self.logger.info(f"Problematic steps content: Completion Steps: {completion_steps}, Reference Steps: {reference_steps}")
                return 0.0

            def dtw_cost_func_numpy(u_np, v_np):
                u_torch = torch.from_numpy(u_np).float().to(self.similarity_checker.device)
                v_torch = torch.from_numpy(v_np).float().to(self.similarity_checker.device)
                cost = 1.0 - torch.nn.functional.cosine_similarity(u_torch.unsqueeze(0), v_torch.unsqueeze(0)).item()
                # self.logger.debug(f"DTW_COST_FUNC: u_np[:3]={u_np[:3]}, v_np[:3]={v_np[:3]}, cost={cost:.4f}") # This is too verbose
                return cost

            if np_comp_embeddings.shape[0] == 0 or np_ref_embeddings.shape[0] == 0:
                self.logger.info("DTW: One of the numpy embedding arrays has 0 steps after processing. Returning 0.0 DTW reward.")
                return 0.0

            list_comp_embeddings_np = [row for row in np_comp_embeddings] # Still needed for len(x) by dtw
            list_ref_embeddings_np = [row for row in np_ref_embeddings]   # Still needed for len(y) by dtw

            # Manually compute the cost matrix
            N = np_comp_embeddings.shape[0]
            M = np_ref_embeddings.shape[0]
            
            cost_matrix = np.zeros((N, M))
            self.logger.debug(f"DTW: Manually creating cost matrix of shape ({N}, {M})")
            for i in range(N):
                for j in range(M):
                    # dtw_cost_func_numpy expects 1D numpy arrays
                    cost_matrix[i, j] = dtw_cost_func_numpy(np_comp_embeddings[i], np_ref_embeddings[j])
            
            self.logger.debug(f"DTW: Manually computed cost_matrix (sample norms): sum={np.sum(cost_matrix):.4f}, mean={np.mean(cost_matrix):.4f}")

            # Call dtw with the precomputed cost matrix.
            # x and y are still passed as dtw-python uses their lengths for pathfinding,
            # but dist_method will be ignored if cost_matrix is provided.
            alignment = dtw(list_comp_embeddings_np, list_ref_embeddings_np, cost_matrix=cost_matrix)
            
            raw_distance = alignment.distance # This is sum of costs along path
            normalized_distance = alignment.normalizedDistance
            self.logger.info(f"DTW: Raw distance = {raw_distance:.4f}, Normalized distance = {normalized_distance:.4f}")

            # Update stats
            self.stats.dtw_stats['total_dtw_distance'] += normalized_distance
            self.stats.dtw_stats['dtw_comparisons_count'] += 1
            if self.stats.dtw_stats['dtw_comparisons_count'] > 0:
                self.stats.dtw_stats['average_dtw_distance'] = \
                    self.stats.dtw_stats['total_dtw_distance'] / self.stats.dtw_stats['dtw_comparisons_count']

            # Convert distance to reward (higher reward for lower distance)
            dtw_reward = self.dtw_max_reward * (1.0 - normalized_distance)
            dtw_reward = max(0.0, dtw_reward) # Ensure reward is not negative
            self.logger.info(f"DTW: Calculated reward component = {dtw_reward:.4f}")
            return dtw_reward

        except Exception as e:
            self.logger.error(f"DTW: Error during DTW calculation: {e}")
            self.logger.debug(f"DTW Error - Completion Steps during error: {completion_steps}")
            self.logger.debug(f"DTW Error - Reference Steps during error: {reference_steps}")
            self.logger.debug(f"DTW Error - np_comp_embeddings shape: {np_comp_embeddings.shape if 'np_comp_embeddings' in locals() else 'not defined'}")
            self.logger.debug(f"DTW Error - np_ref_embeddings shape: {np_ref_embeddings.shape if 'np_ref_embeddings' in locals() else 'not defined'}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0

    async def calculate_reward(self, completion: str, **kwargs) -> float:
        total_reward = 0.0
        
        # For plurality voting and basic answer stats
        batch_index = kwargs.get('reward_index', 0) # Ensure a default
        self._ensure_batch_lists_length(batch_index + 1) # Make sure lists are long enough

        # 1. Basic Correctness Reward (similar to SolutionReward)
        correctness_reward_component = 0.0
        model_answer_numeric = None
        is_final_answer_correct = False

        reference_correct_answer_str = str(kwargs.get('answer', '')) # 'answer' is the ground truth
        
        # Extract final answer from completion
        # Assuming the structure <thinking>...</thinking>\n<solution>...</solution>\n\\boxed{ans}
        # or similar, and extract_answer_from_solution handles it.
        response_content_match = re.search(r"<solution>(.*?)</solution>", completion, re.DOTALL)
        response_content = response_content_match.group(1) if response_content_match else completion
        
        model_answer_str = extract_answer_from_solution(response_content)

        if model_answer_str is not None:
            model_answer_numeric, _ = extract_numeric_answer(model_answer_str)
            correct_answer_numeric, _ = extract_numeric_answer(reference_correct_answer_str)

            if model_answer_numeric is not None and correct_answer_numeric is not None:
                if abs(model_answer_numeric - correct_answer_numeric) <= self.config.numeric_tolerance:
                    correctness_reward_component = self.correctness_reward_value
                    is_final_answer_correct = True
                    self.stats.group_stats['correct_answers'] +=1
                else:
                    self.stats.group_stats['incorrect_answers'] +=1
            else: # Could not parse numeric answers
                 self.stats.group_stats['incorrect_answers'] +=1
        else: # No answer found in model completion
            self.stats.group_stats['incorrect_answers'] +=1

        total_reward += correctness_reward_component
        if correctness_reward_component > 0:
             current_base_correctness = self.stats.reward_components.get('base_correctness_rewards', 0.0)
             self.stats.reward_components['base_correctness_rewards'] = current_base_correctness + correctness_reward_component

        # Store for plurality stats
        self.stats.current_batch['answers'][batch_index] = model_answer_numeric
        self.stats.current_batch['is_correct'][batch_index] = is_final_answer_correct
        self.stats.current_batch['code_lengths'][batch_index] = len(completion) # or len(response_content)
        self.stats.current_batch['completions'][batch_index] = completion


        # 2. Step Count Matching Reward & DTW Similarity Reward
        dtw_reward_component = 0.0
        step_count_reward_component = 0.0
        reference_solution_text = kwargs.get('solution', '') # 'solution' is the reference solution text in dataset

        if reference_solution_text:
            completion_steps = self._extract_steps(completion)
            reference_steps = self._extract_steps(reference_solution_text)
            
            len_comp_steps = len(completion_steps)
            len_ref_steps = len(reference_steps)

            self.logger.info(f"[Steps Counts] Completion: {len_comp_steps}, Reference: {len_ref_steps}")

            # Calculate Step Count Matching Reward
            diff_steps = abs(len_comp_steps - len_ref_steps)
            step_count_reward_component = self.step_count_match_max_reward / (1.0 + float(diff_steps))
            total_reward += step_count_reward_component
            current_sc_reward = self.stats.reward_components.get('step_count_match_rewards', 0.0)
            self.stats.reward_components['step_count_match_rewards'] = current_sc_reward + step_count_reward_component
            self.logger.info(f"Step Count Match: Comp={len_comp_steps}, Ref={len_ref_steps}, Diff={diff_steps}, RewardComp={step_count_reward_component:.4f}")

            # Update step count stats
            self.stats.step_count_stats['total_step_diff'] += diff_steps
            self.stats.step_count_stats['num_step_comparisons'] += 1
            if diff_steps == 0:
                self.stats.step_count_stats['perfect_step_count_matches'] += 1
            if self.stats.step_count_stats['num_step_comparisons'] > 0:
                self.stats.step_count_stats['average_step_diff'] = \
                    self.stats.step_count_stats['total_step_diff'] / self.stats.step_count_stats['num_step_comparisons']


            # Calculate DTW Similarity Reward (only if both have steps)
            if completion_steps and reference_steps:
                self.stats.dtw_stats['completions_with_steps'] += 1
                dtw_reward_component = await self._calculate_dtw_reward_component(completion_steps, reference_steps)
                total_reward += dtw_reward_component
                if dtw_reward_component > 0: # dtw_reward_component is already max(0, val)
                    current_dtw_similarity = self.stats.reward_components.get('dtw_similarity_rewards', 0.0)
                    self.stats.reward_components['dtw_similarity_rewards'] = current_dtw_similarity + dtw_reward_component
            elif completion_steps and not reference_steps:
                self.logger.info("DTW: Completion has steps, but reference solution does not. No DTW reward calculated.")
            elif not completion_steps and reference_steps:
                self.logger.info("DTW: Reference solution has steps, but completion does not. No DTW reward calculated.")
            # If neither has steps, dtw_reward_component remains 0.0
        else:
            self.logger.info("DTW: No reference solution text provided. Skipping Step Count and DTW rewards.")


        # total_rewards is typically managed by RewardStats or BaseReward based on all components.
        # Let's ensure it's updated if not handled by superclass logic based on individual components.
        # self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + total_reward
        # BaseReward.__call__ calls self.stats.update(rewards, ...), which should handle total and average.

        return total_reward

    def _ensure_batch_lists_length(self, required_length: int):
        """Ensure current_batch lists are long enough."""
        for key in ['answers', 'is_correct', 'execution_times', 'code_lengths', 'completions']:
            if key not in self.stats.current_batch:
                self.stats.current_batch[key] = []
            
            current_len = len(self.stats.current_batch[key])
            if current_len < required_length:
                self.stats.current_batch[key].extend([None] * (required_length - current_len))

    # Override _finalize_batch from BaseReward if custom plurality logic is needed,
    # or ensure BaseReward._finalize_batch works with the stats populated by this class.
    # The current BaseReward._finalize_batch should work if 'answers' and 'is_correct' are populated.
    # We added self.answer_grouping_tolerance for it.
