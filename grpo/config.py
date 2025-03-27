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
    
    # Model settings (from BenchmarkConfig)
    main: str = "LOCAL_0"  # Main model to use for solving problems
    main_template: int = 1  # Main model to use for solving problems
    main_port: int = 8007  # Port for main model server
    main_temp: float = 0  # Temperature for main model generation
    
    # Completion reward parameters
    step_continuity_reward: float = 0.5
    
    # Embedding model settings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # Model with larger context support
    embedding_max_length: int = 512  # Increased context length
    embedding_device: str = "auto"  # Use GPU when available
    embedding_batch_size: int = 8  # Adjusted batch size for larger model
    embedding_fallback_to_cpu: bool = True  # Fallback to CPU if GPU fails
    embedding_compute_on_cpu: bool = False # IMPORTANT: Use GPU for performance
    
    # Common reward components
    base_reward: float = 3.0
    validation_reward: float = 0.3
    length_penalty_factor: float = 0.00001
    
    # Solution-specific settings
    solution_validation_reward: float = 0.2  # For passing solution validation
    
    # Programming reward values
    syntax_reward: float = 0.2     # Reward for code with no syntax errors
    syntax_penalty: float = 0.2    # Penalty for code with syntax errors
    execution_reward: float = 0.5  # Reward for code that executes and returns a float
    correctness_reward: float = 2.5 # Reward for code that returns the correct answer
    
    # Tutor-specific settings
    tutor_verdict_reward: float = 0.8  # For correct verdict
    tutor_fix_reward: float = 2.0      # For correct fix
    tutor_combined_reward: float = 3.0 # For correct verdict and fix
    
    # Group-specific settings
    group_diversity_bonus: float = 1
    group_similarity_threshold: float = 0.8

