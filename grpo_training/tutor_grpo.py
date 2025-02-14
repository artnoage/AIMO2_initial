import os
import wandb
import logging
from typing import List
from datasets import load_dataset, load_from_disk
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth.chat_templates import get_chat_template
import sys
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import TutorReward

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('tutor_grpo')
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
    def __init__(self, reward_func, save_frequency=100):
        self.reward_func = reward_func
        self.save_frequency = save_frequency
        self.step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        
        # Log to wandb
        if logs:
            # Add reward function specific metrics
            if 'rewards/0' in logs:
                # Calculate non-accumulative rewards for this round
                current_rewards = {
                    'tutor_reward': logs['rewards/0'],
                    'base_rewards': self.reward_func.stats.reward_components.get('base_rewards', 0) - getattr(self, '_last_base_rewards', 0),
                    'analysis_rewards': self.reward_func.stats.reward_components.get('analysis_rewards', 0) - getattr(self, '_last_analysis_rewards', 0),
                    'substitution_rewards': self.reward_func.stats.reward_components.get('substitution_rewards', 0) - getattr(self, '_last_substitution_rewards', 0),
                    'step_bonuses': self.reward_func.stats.reward_components.get('step_bonuses', 0) - getattr(self, '_last_step_bonuses', 0),
                    'step_penalties': self.reward_func.stats.reward_components.get('step_penalties', 0) - getattr(self, '_last_step_penalties', 0)
                }
                
                # Add accuracy metrics
                total_predictions = self.reward_func.stats.accuracy_stats['total_predictions']
                step_predictions = self.reward_func.stats.accuracy_stats['step_predictions']
                
                accuracy_metrics = {
                    'overall_accuracy': self.reward_func.stats.accuracy_stats['correct_predictions'] / max(1, total_predictions),
                    'conditional_step_accuracy': self.reward_func.stats.accuracy_stats['correct_step_predictions'] / max(1, step_predictions)
                }
                current_rewards.update(accuracy_metrics)
                
                # Store current values for next round
                self._last_base_rewards = self.reward_func.stats.reward_components.get('base_rewards', 0)
                self._last_analysis_rewards = self.reward_func.stats.reward_components.get('analysis_rewards', 0)
                self._last_substitution_rewards = self.reward_func.stats.reward_components.get('substitution_rewards', 0)
                self._last_step_bonuses = self.reward_func.stats.reward_components.get('step_bonuses', 0)
                self._last_step_penalties = self.reward_func.stats.reward_components.get('step_penalties', 0)
                
                wandb.log(current_rewards)
            wandb.log(logs)

def main():
    # Configuration
    model_type = "tutor"
    model_name = "/Home/stat/laschos/AIMO2_initial/models/tutor/20250210_064759"
    dataset_path = "/Home/stat/laschos/AIMO2_initial/local_datasets/tutor_training/20250211_084032"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    try:
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Initialize config
    reward_config = RewardConfig(model_type=model_type)
    
    # Setup
    logger = setup_logging(reward_config.model_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"tutor with sfted model {timestamp}"
    # Initialize wandb
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": model_type,
            "dataset": dataset_path,
            "structure_base_reward": 0.2,
            "analysis_reward": 0.2,
            "substitution_reward": 0.4,
            "single_step_bonus": 0.2,
            "multiple_step_penalty": 0.4,
            "full_reward": 5.0
        }
    )
    
    # Initialize reward function with existing config
    reward_func = TutorReward(reward_config)
    

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # Use the model_name variable defined at the start
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_lora_rank=128
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    
    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True
    )
    
        
    def formatting_func(example):
        formatted_example = {**example}
        formatted_example["prompt"] = f"[INST]{example['prompt']}[/INST]"
        return formatted_example
    
    formatted_dataset = dataset.map(
        formatting_func,
        desc="Applying chat template"
    )
    
    # Training arguments
    training_args = GRPOConfig(
        use_vllm=True,
        torch_empty_cache_steps=10,
        learning_rate=3e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=10,
        max_prompt_length=3000,
        max_completion_length=1096,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
        gradient_checkpointing=True,
        report_to="wandb",
        output_dir=output_dir,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback(reward_func=reward_func, save_frequency=10)]
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
