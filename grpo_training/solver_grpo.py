import os
import sys
import wandb
import logging
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import TrainerCallback
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer, validate_solution
from .config import GRPOConfig as RewardConfig
from .rewards import SolutionReward

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('solver_grpo')
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
        
        # Log to wandb
        if logs:
            # Add reward function specific metrics
            if 'rewards/0' in logs:
                # Print rewards grouped by 8 for better visualization
                rewards = logs['rewards/0']
                print("\nAll rewards (grouped by 8):")
                for i in range(0, len(rewards), 8):
                    group = rewards[i:i+8]
                    print(f"Group {i//8}: {[round(r, 3) for r in group]}")
                
                wandb.log({
                    'correctness_reward': logs['rewards/0'],
                    'total_rewards': self.reward_func.stats.total_rewards,
                    'avg_reward': self.reward_func.stats.total_rewards / max(1, self.reward_func.stats.total_batches),
                    'base_rewards': self.reward_func.stats.reward_components.get('base_rewards', 0),
                    'validation_rewards': self.reward_func.stats.reward_components.get('validation_rewards', 0),
                    'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
                })
            wandb.log(logs)
            
            # Log detailed statistics periodically
            if self.step % self.save_frequency == 0:
                self.logger.info(f"\nDetailed Statistics at step {self.step}:")
                self.logger.info(f"Total batches processed: {self.reward_func.stats.total_batches}")
                self.logger.info(f"Average reward: {self.reward_func.stats.total_rewards / max(1, self.reward_func.stats.total_batches):.4f}")
                self.logger.info(f"Base rewards given: {self.reward_func.stats.reward_components.get('base_rewards', 0)}")
                self.logger.info(f"Validation rewards given: {self.reward_func.stats.reward_components.get('validation_rewards', 0)}")
                self.logger.info(f"Total length penalty: {self.reward_func.stats.reward_components.get('total_length_penalty', 0.0):.4f}")

def main():
    # Initialize config
    reward_config = RewardConfig(model_type="solver")
    
    # Setup
    logger = setup_logging(reward_config.model_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project="solver_grpo",
        name=f"solver_grpo_{timestamp}",
        config={
            "model_type": reward_config.model_type,
            "dataset": reward_config.dataset_name,
            "base_reward": 2.0,
            "validation_reward": 0.2,
            "length_penalty_factor": 0.0001
        }
    )
    
    # Initialize reward function
    reward_func = SolutionReward(reward_config)
    
    # Load model
    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=reward_config.model_name,
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_lora_rank=64
    )
    
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
    
    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True
    )
    
    # Load and format dataset
    dataset = load_dataset(reward_config.dataset_name)
    def formatting_func(example):
        solver_prompt = (
            "Here is a mathematical problem:\n\n"
            f"{example['problem']}\n\n"
            "Could you help me solve this from start to finish? First, let's analyze the problem, "
            "then walk through the solution step-by-step using LaTeX notation. "
            "Don't forget to put the final answer in a box using \\boxed{}"
        )
        return {
            "prompt": f"[INST]{solver_prompt}[/INST]",
            "answer": example['answer']
        }
    
    formatted_dataset = dataset['train'].map(
        formatting_func,
        desc="Applying chat template"
    )
    
    # Print first entry tokenization
    first_entry = formatted_dataset[0]
    print("\nFirst entry tokenization:")
    print("Original:", first_entry['prompt'])
    tokenized = tokenizer(first_entry['prompt'])
    print("Tokenized:", tokenized)
    print("Decoded:", tokenizer.decode(tokenized['input_ids']))
    
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
        per_device_train_batch_size=3,
        gradient_accumulation_steps=1,
        num_generations=5,
        max_prompt_length=1348,
        max_completion_length=5148,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
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
        callbacks=[LoggingCallback(reward_func=reward_func, logger=logger, save_frequency=100)]
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
