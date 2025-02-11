from dataclasses import dataclass
from typing import Optional

@dataclass
class GRPOConfig:
    """Configuration for GRPO training and rewards"""
    
    # Model settings
    model_type: str
    completion_model_name: str = "/Home/stat/laschos/AIMO2_initial/models/light/20250209_172917"
    
    # Model size settings
    max_seq_length: int = 4096
    max_lora_rank: int = 64
    lora_alpha: int = 64
    
    # Embedding model settings
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_max_length: int = 512
    
    # API settings
    completion_port: int = 8004
    completion_attempts: int = 10
    
    # Common reward settings
    numeric_tolerance: float = 1e-6
    logging_dir: str = "logs"
    stats_dir: str = "statistics"
    
    # Basic solution rewards
    solution_base_reward: float = 2.0
    solution_validation_reward: float = 0.2
    solution_length_penalty_factor: float = 0.0001
    
    # Group rewards
    group_base_reward: float = 3.0
    group_diversity_bonus: float = 0.3
    group_majority_bonus: float = 0.2
    group_similarity_threshold_low: float = 0.7
    group_similarity_threshold_high: float = 0.9
    
    # Tutor rewards
    tutor_structure_base_reward: float = 0.2
    tutor_analysis_reward: float = 0.2
    tutor_substitution_reward: float = 0.4
    tutor_single_step_bonus: float = 0.2
    tutor_multiple_step_penalty: float = 0.4
    tutor_full_reward: float = 5.0
    tutor_analysis_length_cost: float = 0.0001
    tutor_substitution_length_cost: float = 0.0001
    tutor_redundant_substitution_penalty: float = 0.1
    tutor_wrong_boxed_answer_penalty: float = 1.0
