import os
import wandb
import logging
import torch
import re
import asyncio
import sys
from datasets import load_dataset, Dataset
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Any, Union

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import BaseReward
from reward_stats import RewardStats
from utils.solution_utils import extract_numeric_answer
from utils.model_utils import time_limit

# Import system prompts from agents.py
from utils.agents import PROGRAMMER_SYSTEM_PROMPT

# Use the system prompt from agents.py
SYSTEM_PROMPT = PROGRAMMER_SYSTEM_PROMPT

class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('programming_grpo')
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
            # Key performance metrics for wandb
            wandb_stats = {
                'programming_reward': logs['rewards/0'],
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0),
                'correct_solutions': self.reward_func.stats.reward_components.get('correct_solutions', 0),
                'syntax_valid_solutions': self.reward_func.stats.reward_components.get('syntax_valid_solutions', 0),
                'execution_valid_solutions': self.reward_func.stats.reward_components.get('execution_valid_solutions', 0)
            }
            
            # Detailed stats for local logging only
            local_stats = {
                'reward_components': {
                    'structure_rewards': self.reward_func.stats.reward_components.get('structure_rewards', 0) - getattr(self, '_last_structure_rewards', 0),
                    'syntax_rewards': self.reward_func.stats.reward_components.get('syntax_rewards', 0) - getattr(self, '_last_syntax_rewards', 0),
                    'execution_rewards': self.reward_func.stats.reward_components.get('execution_rewards', 0) - getattr(self, '_last_execution_rewards', 0),
                    'correctness_rewards': self.reward_func.stats.reward_components.get('correctness_rewards', 0) - getattr(self, '_last_correctness_rewards', 0),
                    'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0) - getattr(self, '_last_length_penalties', 0.0)
                }
            }
            
            # Store current values for next round
            self._last_structure_rewards = self.reward_func.stats.reward_components.get('structure_rewards', 0)
            self._last_syntax_rewards = self.reward_func.stats.reward_components.get('syntax_rewards', 0)
            self._last_execution_rewards = self.reward_func.stats.reward_components.get('execution_rewards', 0)
            self._last_correctness_rewards = self.reward_func.stats.reward_components.get('correctness_rewards', 0)
            self._last_length_penalties = self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
            
            # Update logs with our metrics
            logs.update(wandb_stats)

# These functions are now imported from rewards.py
from rewards import extract_code_from_response, check_code_quality, run_code_safely

# Use the ProgrammingReward class from rewards.py
from rewards import ProgrammingReward

def main():
    # Configuration
    model_type = "programming_0"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/qwen_sft/20250303_224627"
    dataset_name = "Metaskepsis/Numina_hard_filtered"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Initialize config with reward values
    reward_config = RewardConfig(model_type=model_type)
    # Reward values are already set in the config class
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"{model_type}, {model_name}, {dataset_name}, {timestamp}"
    
    # Initialize wandb
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": reward_config.model_type,
            "dataset": dataset_name,
            "structure_reward": reward_config.structure_reward,
            "syntax_reward": reward_config.syntax_reward,
            "execution_reward": reward_config.execution_reward,
            "correctness_reward": reward_config.correctness_reward
        }
    )
    
    # Initialize reward function
    reward_func = ProgrammingReward(reward_config)
    logger.info("\nInitialized ProgrammingReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=0.6,
        max_lora_rank=64)
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    
    def get_questions(split="train") -> Dataset:
        # Import the data preparation function
        from utils.data_preparation import prepare_programming_data
        
        # Load dataset
        data = load_dataset(dataset_name, split=split)
        return prepare_programming_data(data, SYSTEM_PROMPT, split)
    
    formatted_dataset = get_questions()
    formatted_dataset = formatted_dataset.shuffle(seed=42)
    # Use a smaller dataset for programming training
    formatted_dataset = formatted_dataset.select(range(1000))
    
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Answer: {entry.get('answer')}")
        print(f"Correct answer: {entry.get('correct_answer')}")
    
    # GRPO specific training arguments
    training_args = GRPOConfig(
        torch_empty_cache_steps=1,
        learning_rate=6e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_generations=4,  # Fewer generations for programming tasks
        max_prompt_length=800,
        max_completion_length=3296,
        num_train_epochs=1,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
    )
    
    # Initialize trainer with reward function
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback(reward_func=reward_func, logger=logger, save_frequency=10)]
    )
    
    # Train
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        wandb.finish()
        raise
        
    # Save model
    try:
        models_dir = "models"
        os.makedirs(os.path.join(models_dir, reward_config.model_type), exist_ok=True)
        model_output_dir = os.path.join(models_dir, reward_config.model_type, timestamp)
        model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"Merged model saved to {model_output_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
