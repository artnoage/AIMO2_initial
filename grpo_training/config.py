from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Base configuration for reward calculation"""
    model_type: str
    numeric_tolerance: float = 1e-6
    logging_dir: str = "logs"
    stats_dir: str = "statistics"
    max_retries: int = 3
    timeout: int = 300
    
    # Completion agent settings
    completion_port: int = 8004
    completion_temp: float = 0.7
    completion_attempts: int = 10
    
    # Embedding model settings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_max_length: int = 512
    
    # Common reward components
    base_reward: float = 3.0
    validation_reward: float = 0.3
    length_penalty_factor: float = 0.0001
    
    # Solution-specific settings
    solution_base_reward: float = 0.05  # Initial reward
    solution_reasoning_reward: float = 0.1  # For having reasoning section
    solution_response_reward: float = 0.1  # For having response section
    solution_steps_reward: float = 0.05  # For having numbered steps
    solution_ordered_steps_reward: float = 0.05  # Additional for ordered steps
    
    # Group-specific settings
    group_base_reward: float = 3
    group_diversity_bonus: float = 0.3
    group_majority_bonus: float = 0.2
    group_shortest_bonus: float = 0.2  # Bonus for shortest correct solution
    group_similarity_threshold_low: float = 0.7
    group_similarity_threshold_high: float = 0.9
    
    # Tutor-specific settings
    tutor_structure_base_reward: float = 0.2
    tutor_analysis_reward: float = 0.2
    tutor_substitution_reward: float = 0.4
    tutor_single_step_bonus: float = 0.2
    tutor_multiple_step_penalty: float = 0.4
    tutor_full_reward: float = 5.0
    tutor_analysis_length_cost: float = 0.0001
    tutor_substitution_length_cost: float = 0.0001
    tutor_redundant_substitution_penalty: float = 0.1
