"""
Multi-GPU Training for GRPO 7b using DeepSpeed ZeRO Stage 3

This script demonstrates how to train a GRPO 7b model using DeepSpeed ZeRO Stage 3
for efficient multi-GPU training. It's based on the user's original code but modified
to use DeepSpeed instead of DataParallel for better GPU utilization.

Usage:
    deepspeed --num_gpus=4 grpo_7b_multi_gpu_training.py
"""

import os
import wandb
import logging
from datasets import load_dataset, concatenate_datasets, Dataset
from datetime import datetime
import sys
import torch
from functools import partial
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer
import deepspeed

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your custom modules
# Adjust these imports based on your actual project structure
try:
    from config import RewardConfig
    from dynamic_reward import DynamicReward
    from utils.similarity_checker import SolutionSimilarityChecker
    from utils.data_preparation import prepare_combined_data
    from utils.agents import (
        FULLSOLUTION_SYSTEM_PROMPT, 
        FINALIZATION_SYSTEM_PROMPT,
        PROGRAMMER_SYSTEM_PROMPT,
        TUTOR_SYSTEM_PROMPT,
        TESTER_SYSTEM_PROMPT,
        ARCHITECT_SYSTEM_PROMPT
    )
except ImportError:
    print("Warning: Could not import custom modules. This script assumes they exist in your project.")
    # Define placeholder classes/functions for demonstration purposes
    class RewardConfig:
        def __init__(self, model_type):
            self.model_type = model_type
            self.group_diversity_bonus = 0.3
    
    class DynamicReward:
        def __init__(self, config, checker):
            self.config = config
            self.checker = checker
            self.stats = type('obj', (object,), {
                'reward_components': {},
                'group_stats': {},
                'step_stats': {},
                'similarity_stats': {},
                'programming_stats': {},
                'get_summary': lambda: "Stats summary"
            })
    
    class SolutionSimilarityChecker:
        def __init__(self, config):
            self.config = config
    
    def prepare_combined_data(*args, **kwargs):
        return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    
    FULLSOLUTION_SYSTEM_PROMPT = "You are a helpful assistant."
    FINALIZATION_SYSTEM_PROMPT = "You are a helpful assistant."
    PROGRAMMER_SYSTEM_PROMPT = "You are a helpful assistant."
    TUTOR_SYSTEM_PROMPT = "You are a helpful assistant."
    TESTER_SYSTEM_PROMPT = "You are a helpful assistant."
    ARCHITECT_SYSTEM_PROMPT = "You are a helpful assistant."

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('dynamic_grpo')
    
    # Clear any existing handlers to prevent duplicate logging
    if logger.handlers:
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        f"{log_dir}/training_{timestamp}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    return logger

class LoggingCallback(TrainerCallback):
    """Callback for logging training metrics"""
    def __init__(self, reward_func, logger, save_frequency=100):
        self.reward_func = reward_func
        self.save_frequency = save_frequency
        self.step = 0
        self.logger = logger
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        
        if logs and 'rewards/0' in logs and hasattr(self.reward_func, 'stats'):
            # Print detailed stats to console/log file
            self.logger.info("\n" + "="*50)
            self.logger.info(f"Step {self.step} - Reward Stats Summary:")
            
            # Get and log the stats summary
            stats_summary = self.reward_func.stats.get_summary()
            self.logger.info(stats_summary)
            self.logger.info("="*50 + "\n")
            
            # Key performance metrics for wandb
            wandb_stats = {
                'reward': logs['rewards/0'],
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0),
                'total_batches': self.reward_func.stats.total_batches,
                'total_examples': self.reward_func.stats.total_examples
            }
            
            # Add dynamic reward specific metrics
            if 'solution_reward_uses' in self.reward_func.stats.reward_components:
                wandb_stats['solution_reward_uses'] = self.reward_func.stats.reward_components['solution_reward_uses']
            if 'completion_reward_uses' in self.reward_func.stats.reward_components:
                wandb_stats['completion_reward_uses'] = self.reward_func.stats.reward_components['completion_reward_uses']
                
            # Track example types in the batch
            if hasattr(state, 'train_dataloader') and state.train_dataloader is not None:
                try:
                    # Get current batch
                    batch_idx = (state.global_step - 1) % len(state.train_dataloader)
                    current_batch = list(state.train_dataloader)[batch_idx]
                    
                    # Count example types if available
                    if 'example_type' in current_batch:
                        example_types = current_batch['example_type']
                        solution_count = sum(1 for t in example_types if t == 'solution')
                        completion_count = sum(1 for t in example_types if t == 'completion')
                        wait_count = sum(1 for t in example_types if t == 'wait')
                        
                        wandb_stats['solution_examples'] = solution_count
                        wandb_stats['completion_examples'] = completion_count
                        wandb_stats['wait_examples'] = wait_count
                except Exception as e:
                    self.logger.warning(f"Could not track example types: {str(e)}")
            
            # Add all stats from reward_components to wandb
            for key, value in self.reward_func.stats.reward_components.items():
                wandb_stats[f'reward_components/{key}'] = value
                
            # Add group stats
            for key, value in self.reward_func.stats.group_stats.items():
                wandb_stats[f'group_stats/{key}'] = value
                
            # Add step stats
            for key, value in self.reward_func.stats.step_stats.items():
                wandb_stats[f'step_stats/{key}'] = value
                
            # Add similarity stats
            for key, value in self.reward_func.stats.similarity_stats.items():
                wandb_stats[f'similarity_stats/{key}'] = value
                
            # Add programming stats
            for key, value in self.reward_func.stats.programming_stats.items():
                wandb_stats[f'programming_stats/{key}'] = value
                
            # Add reward distribution
            if hasattr(self.reward_func.stats, 'reward_distribution') and self.reward_func.stats.reward_distribution:
                # Only log the top 10 most common rewards to avoid cluttering wandb
                sorted_rewards = sorted(
                    self.reward_func.stats.reward_distribution.items(), 
                    key=lambda x: self.reward_func.stats.reward_distribution[x[0]], 
                    reverse=True
                )[:10]
                
                for reward, count in sorted_rewards:
                    wandb_stats[f'reward_distribution/{reward}'] = count
            
            # Update logs with our metrics
            logs.update(wandb_stats)

def log_gpu_usage(logger):
    """Log GPU memory usage during training"""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available. Using CPU only.")
        return
    
    logger.info("GPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1e9
        memory_reserved = torch.cuda.memory_reserved(i) / 1e9
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"  Memory Allocated: {memory_allocated:.2f} GB")
        logger.info(f"  Memory Reserved: {memory_reserved:.2f} GB")

def main():
    print("Starting script execution...")
    
    # Configuration
    model_type = "dynamic_0"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/dynamic_2/20250324_215025"  # Update with your model path
    dataset_name = "Metaskepsis/Olympiads_medium"
    
    # Setup logging first
    logger = setup_logging(model_type)
    logger.info("Logging initialized")
    
    # Initialize config
    reward_config = RewardConfig(model_type=model_type)
    logger.info("Reward config initialized")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"{model_type}, DB={reward_config.group_diversity_bonus}, {model_name}, {dataset_name}, {timestamp}"
    
    # Initialize wandb
    logger.info("Initializing wandb...")
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": reward_config.model_type,
            "dataset": dataset_name,
            "base_reward": 3.0,
            "diversity_bonus": 0.3,
            "step_continuity_reward": 0.5
        }
    )
    logger.info("Wandb initialized")
    
    # Initialize similarity checker first
    logger.info("Initializing similarity checker...")
    similarity_checker = SolutionSimilarityChecker(reward_config)
    logger.info("Similarity checker initialized")
    
    # Initialize dynamic reward function
    logger.info("Initializing dynamic reward function...")
    reward_func = DynamicReward(reward_config, similarity_checker)
    logger.info("\nInitialized DynamicReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    
    # Print initial stats configuration
    if hasattr(reward_func, 'stats'):
        logger.info("Initial stats configuration:")
        for category in ['reward_components', 'group_stats', 'step_stats', 'similarity_stats']:
            if hasattr(reward_func.stats, category):
                stats_dict = getattr(reward_func.stats, category)
                logger.info(f"{category}: {stats_dict}")
    else:
        logger.warning("No stats object found in reward_func!")
    
    # Check GPU availability
    logger.info("Checking GPU availability...")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPUs available. Training will be slow on CPU.")
        num_gpus = 0
    
    # Load model using Hugging Face Transformers
    logger.info(f"Loading model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded")
    
    # Load model with DeepSpeed compatibility
    # Note: We don't specify device_map to allow DeepSpeed to manage device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    logger.info("Model loaded")
    
    def get_questions(split="train") -> Dataset:
        """Load and format dataset with full solution, completion, programming, and wait examples
        with the following distribution:
        - 35% solution examples
        - 35% programming examples
        - 15% completion examples
        - 15% wait examples
        """
        logger.info("Loading and preparing dataset...")
        
        # Load the base dataset
        data1 = load_dataset(dataset_name, split="train")
        data1 = data1.shuffle(seed=141)
        data1 = data1.select(range(2500))
        data2 = load_dataset("Metaskepsis/Olympiads_hard", split="train")
        data2 = data2.select(range(500))
        data2 = data2.shuffle(seed=141)
        data = concatenate_datasets([data1, data2])
        data = data.shuffle(seed=141)
        logger.info("Dataset loaded and shuffled")
        
        # Define the distribution
        # You can set any value to 0 to skip generating that type of example
        distribution = {
            'solution': 0.25,
            'programming': 0.25,
            'finalization': 0,
            'tutor': 0,
            'test_programming': 0.25,
            'architect': 0.25
        }
        
        # Use the prepare_combined_data function with all system prompts
        logger.info("Preparing combined data with system prompts...")
        return prepare_combined_data(
            data, 
            FULLSOLUTION_SYSTEM_PROMPT, 
            FINALIZATION_SYSTEM_PROMPT, 
            PROGRAMMER_SYSTEM_PROMPT,
            TUTOR_SYSTEM_PROMPT,
            TESTER_SYSTEM_PROMPT,
            ARCHITECT_SYSTEM_PROMPT,
            tokenizer, 
            distribution)

    # Get the formatted dataset with all types of examples
    logger.info("Getting formatted dataset...")
    formatted_dataset = get_questions()
    logger.info("Formatted dataset created")
    
    # Create DeepSpeed config file if it doesn't exist
    ds_config_path = "ds_config.json"
    if not os.path.exists(ds_config_path):
        logger.info("Creating DeepSpeed config file...")
        
        # Create config in DeepSpeed format
        import json
        ds_config = {
            "fp16": {
                "enabled": "auto",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": "auto"
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 2e-6,
                    "betas": [0.9, 0.99],
                    "eps": 1e-8,
                    "weight_decay": 0.1
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-6,
                    "warmup_num_steps": 100,
                    "total_num_steps": 1000
                }
            },
            "gradient_accumulation_steps": 16,
            "gradient_clipping": 0.1,
            "steps_per_print": 10,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": 8,
            "wall_clock_breakdown": False
        }
        
        with open(ds_config_path, 'w') as f:
            json.dump(ds_config, f, indent=4)
        logger.info(f"DeepSpeed config file created at {ds_config_path}")
    
    # GRPO specific training arguments with DeepSpeed integration
    logger.info("Setting up training arguments...")
    training_args = GRPOConfig(
        torch_empty_cache_steps=1,
        # We'll use minimal optimizer settings since DeepSpeed config handles most of it
        learning_rate=2e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  # Must specify a valid optimizer
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        num_generations=8,
        max_prompt_length=1800,
        max_completion_length=5200,
        num_train_epochs=1,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
        
        # DeepSpeed integration
        deepspeed=ds_config_path,
        local_rank=-1  # Will be set by deepspeed launcher
    )
    logger.info("Training arguments set up")
    
    # Log the dataset structure before training
    logger.info("Dataset structure before training:")
    sample_example = formatted_dataset[0]
    for key, value in sample_example.items():
        logger.info(f"  {key}: {type(value)} - {value}")
    
    # Initialize trainer with reward function
    logger.info("Initializing GRPOTrainer...")
    
    # Create the trainer with DeepSpeed integration
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback(reward_func=reward_func, logger=logger, save_frequency=10)]
    )
    logger.info("GRPOTrainer initialized")
    
    # Log GPU memory usage before training
    log_gpu_usage(logger)
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed")
    
    # Log GPU memory usage after training
    log_gpu_usage(logger)
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    # Close wandb
    wandb.finish()
    logger.info("Wandb session closed")
    
    logger.info("Script execution completed successfully")

if __name__ == "__main__":
    main()
