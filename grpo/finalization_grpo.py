import os
import wandb
import logging
import json
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import sys
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import re
import time
from time import time
import random

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import FinalizationReward, SolutionSimilarityChecker
from utils.agents import FINALIZATION_SYSTEM_PROMPT

# Use the system prompt from agents.py
SYSTEM_PROMPT = FINALIZATION_SYSTEM_PROMPT

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('finalization_grpo')
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
                'reward': logs['rewards/0'],
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0),
                'correct_answers': self.reward_func.stats.reward_components.get('correct_answers', 0),
                'step_continuity': self.reward_func.stats.step_stats.get('correct_step_numbering', 0) / 
                                  max(1, self.reward_func.stats.step_stats.get('correct_step_numbering', 0) + 
                                     self.reward_func.stats.step_stats.get('incorrect_step_numbering', 0))
            }
            
            # Detailed stats for local logging only
            local_stats = {
                'reward_components': {
                    'base_rewards': self.reward_func.stats.reward_components.get('base_rewards', 0) - getattr(self, '_last_base_rewards', 0),
                    'step_continuity_rewards': self.reward_func.stats.reward_components.get('step_continuity_rewards', 0) - getattr(self, '_last_step_continuity_rewards', 0),
                    'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0) - getattr(self, '_last_length_penalties', 0.0)
                },
                'step_stats': {
                    'correct_step_numbering': self.reward_func.stats.step_stats.get('correct_step_numbering', 0) - getattr(self, '_last_correct_step_numbering', 0),
                    'incorrect_step_numbering': self.reward_func.stats.step_stats.get('incorrect_step_numbering', 0) - getattr(self, '_last_incorrect_step_numbering', 0),
                    'total_steps_completed': self.reward_func.stats.step_stats.get('total_steps_completed', 0) - getattr(self, '_last_total_steps_completed', 0)
                }
            }
            
            # Store current values for next round
            self._last_base_rewards = self.reward_func.stats.reward_components.get('base_rewards', 0)
            self._last_step_continuity_rewards = self.reward_func.stats.reward_components.get('step_continuity_rewards', 0)
            self._last_length_penalties = self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
            self._last_correct_step_numbering = self.reward_func.stats.step_stats.get('correct_step_numbering', 0)
            self._last_incorrect_step_numbering = self.reward_func.stats.step_stats.get('incorrect_step_numbering', 0)
            self._last_total_steps_completed = self.reward_func.stats.step_stats.get('total_steps_completed', 0)
            
            # Update logs with our metrics
            logs.update(wandb_stats)

def main():
    # Configuration
    model_type = "finalization"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/qwen_sft/20250303_224627"
    dataset_name = "/Home/stat/laschos/math/AIMO2_initial/local_datasets/20250301_141300"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Log dataset information
    logger.info(f"Using dataset: {dataset_name}")
    
    # Initialize config
    reward_config = RewardConfig(model_type=model_type)
    reward_config.step_continuity_reward = 1.0  # Reward for correctly continuing step numbering
    
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
            "base_reward": reward_config.base_reward,
            "step_continuity_reward": reward_config.step_continuity_reward,
            "length_penalty_factor": reward_config.length_penalty_factor
        }
    )
    
    # Initialize similarity checker first
    similarity_checker = SolutionSimilarityChecker(reward_config)
    
    # Initialize reward function with similarity checker
    reward_func = FinalizationReward(reward_config, similarity_checker)
    logger.info("\nInitialized FinalizationReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization= 0.6,
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
    
    # Import the data preparation function
    from utils.data_preparation import prepare_detailed_finalization_data
    
    # Apply the transformation and filter out invalid results
    data = load_from_disk(dataset_name)
    
    # Log dataset structure before processing
    logger.info(f"Original dataset columns: {data.column_names}")
    logger.info(f"Sample example keys: {list(data[0].keys()) if len(data) > 0 else 'No examples'}")
    
    # Process the data using the imported function
    valid_data = prepare_detailed_finalization_data(data, SYSTEM_PROMPT, tokenizer)
    
    # Shuffle and select examples
    valid_data = valid_data.shuffle(seed=11)
    max_examples = min(2000, len(valid_data))
    valid_data = valid_data.select(range(max_examples))
    
    # More debug information
    logger.info(f"Using {len(valid_data)} examples for training")
    logger.info(f"Dataset columns: {valid_data.column_names}")
    logger.info(f"First example prompt length: {len(valid_data[0]['prompt']) if len(valid_data) > 0 else 'N/A'}")
    
    # Verify all examples have the required fields
    missing_fields = []
    for i in range(min(10, len(valid_data))):  # Check first 10 examples
        example = valid_data[i]
        if isinstance(example, dict):
            if not example.get('prompt') or not example.get('problem') or not example.get('partial_solution'):
                missing_fields.append(i)
        else:
            logger.error(f"Example at index {i} is not a dictionary but a {type(example)}")
            missing_fields.append(i)
    
    if missing_fields:
        logger.warning(f"Some examples are missing required fields: {missing_fields}")
        for i in missing_fields:
            if i < len(valid_data):
                example = valid_data[i]
                if isinstance(example, dict):
                    logger.warning(f"Example {i} fields: {list(example.keys())}")
                else:
                    logger.warning(f"Example {i} is not a dictionary: {type(example)}")
    
    
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
        per_device_train_batch_size=9,
        gradient_accumulation_steps=4,
        num_generations=9,
        max_prompt_length=2048,
        max_completion_length=2048,
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
        train_dataset=valid_data,
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
