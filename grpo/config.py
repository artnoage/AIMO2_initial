from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Base configuration for reward calculation"""
    model_type: str
    numeric_tolerance: float = 1e-6
    logging_dir: str = "logs"
    stats_dir: str = "statistics"
    max_retries: int = 3
    timeout: int = 60
    
    # Main model settings
    main: str = "LOCAL_0"  # Main model to use for solving problems
    main_template: int = 1  # Template for main model
    main_port: int = 8000  # Port for main model server
    main_temp: float = 0.7  # Temperature for main model generation

    # Auxiliary model settings
    auxiliary: str = "LOCAL_3"  # Auxiliary model to use for solving problems
    auxiliary_template: int = 1  # Template for auxiliary model
    auxiliary_port: int = 8000  # Port for auxiliary model server
    auxiliary_temp: float =  0.7  # Temperature for auxiliary model generation

    # Secondary auxiliary model settings
    auxiliary2: str = "LOCAL_3"  # Secondary auxiliary model to use for solving problems
    auxiliary2_template: int = 1  # Template for secondary auxiliary model
    auxiliary2_port: int = 8000  # Port for secondary auxiliary model server
    auxiliary2_temp: float =  0.7  # Temperature for secondary auxiliary model generation
    # Completion reward parameters
    step_continuity_reward: float = 0.5
    
    # Common reward components
    base_reward: float = 3.0
    validation_reward: float = 0.3
    
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
    
    # Verification-specific settings
    verification_reward: float = 1.5   # Reward for verified solution


