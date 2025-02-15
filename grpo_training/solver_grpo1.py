import os
import wandb
import logging
from typing import List
from datasets import load_dataset, load_from_disk, concatenate_datasets
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
from rewards import SolutionReward

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
        
        if logs and 'rewards/0' in logs:
            # Key performance metrics for wandb
            wandb_stats = {
                'solver_reward': logs['rewards/0'],
                'correct_ratio': self.reward_func.stats.reward_components.get('correct_answers', 0) / max(1, self.reward_func.stats.total_batches),
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0)
            }
            
            # Detailed stats for local logging only
            local_stats = {
                'correct_answers': self.reward_func.stats.reward_components.get('correct_answers', 0) - getattr(self, '_last_correct_answers', 0),
                'incorrect_answers': self.reward_func.stats.reward_components.get('incorrect_answers', 0) - getattr(self, '_last_incorrect_answers', 0),
                'base_rewards': self.reward_func.stats.reward_components.get('base_rewards', 0) - getattr(self, '_last_base_rewards', 0),
                'validation_rewards': self.reward_func.stats.reward_components.get('validation_rewards', 0) - getattr(self, '_last_validation_rewards', 0),
                'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0) - getattr(self, '_last_length_penalties', 0.0)
            }
            
            # Store current values for next round
            self._last_base_rewards = self.reward_func.stats.reward_components.get('base_rewards', 0)
            self._last_validation_rewards = self.reward_func.stats.reward_components.get('validation_rewards', 0)
            self._last_length_penalties = self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
            self._last_correct_answers = self.reward_func.stats.reward_components.get('correct_answers', 0)
            self._last_incorrect_answers = self.reward_func.stats.reward_components.get('incorrect_answers', 0)
            
            # Update wandb logs
            logs.update(wandb_stats)
            
            # Log detailed statistics periodically
            if self.step % self.save_frequency == 0:
                self.logger.info(f"\nDetailed Statistics at step {self.step}:")
                self.logger.info("\nWandb tracked metrics:")
                for key, value in wandb_stats.items():
                    self.logger.info(f"  {key}: {value}")
                    
                self.logger.info("\nAdditional local metrics:")
                for key, value in local_stats.items():
                    self.logger.info(f"  {key}: {value}")
                    
                self.logger.info("\nAccumulated statistics:")
                self.logger.info(f"Total batches processed: {self.reward_func.stats.total_batches}")
                self.logger.info(f"Total rewards: {self.reward_func.stats.total_rewards}")
                self.logger.info(self.reward_func.stats.get_summary())

def main():
    # Configuration
    model_type = "solver2"
    model_name = "/Home/stat/laschos/AIMO2_initial/models/light/20250209_172917"
    dataset_name = "Metaskepsis/Numina_medium"
    
    # Initialize config
    reward_config = RewardConfig(model_type=model_type)
    
    # Setup logging
    logger = setup_logging(model_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{model_type}/{timestamp}"
    wandbname=f"solver1 good light Numina medium  {timestamp}"
    # Initialize wandb
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": model_type,
            "model":model_name,
            "dataset": dataset_name,
            "base_reward": 2.0,
            "validation_reward": 0.2,
            "length_penalty_factor": 0.0001
        }
    )
    
    reward_func = SolutionReward(reward_config)
    

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # Use the model_name variable defined at the start
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
    
    try:
        if os.path.exists(dataset_name):
            dataset = load_from_disk(dataset_name)
        else:
            dataset = load_dataset(dataset_name, 'main')
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
        
    def formatting_func(example):
        solver_prompt = (
        "Here is a mathematical problem:\n\n"
        f"{example['question']}\n\n"
        "Respond in the following format:\n"
        "<reasoning>...</reasoning><response>...</response>\n\n"
        "In the response, provide the full solution in LaTeX with clear, numbered steps. "
        "Ensure that the final answer is enclosed in a box using \\boxed{}."
    )
        
        
        formatted = {
            "prompt": f"[INST]{solver_prompt}[/INST]",
            "answer": example.get('answer'),
            "correct_answer": example.get('answer')
        }
        
        return formatted
        

    
    # Format dataset and ensure answer field is present
    formatted_dataset = dataset['train'].map(
        formatting_func,
        desc="Applying chat template",
        remove_columns=None  # Keep original columns
    )
    # Take first 3000 entries
    
    formatted_dataset = formatted_dataset.select(range(1000))
    shuffled_dataset = formatted_dataset.shuffle(seed=42)
    shuffled_dataset2=shuffled_dataset.shuffle(seed=42)
    #shuffled_dataset3=shuffled_dataset2.shuffle(seed=42)
    #shuffled_dataset4=shuffled_dataset3.shuffle(seed=42)
    # Concatenate original and shuffled datasets
    formatted_dataset = concatenate_datasets([shuffled_dataset,shuffled_dataset2])
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Answer: {entry.get('answer')}")
        print(f"Correct answer: {entry.get('correct_answer')}")
    
    # Print first entry tokenization
    first_entry = formatted_dataset[0]
    print("\nFirst entry tokenization:")
    print("Original:", first_entry['prompt'])
    tokenized = tokenizer(first_entry['prompt'])
    print("Tokenized:", tokenized)
    print("Decoded:", tokenizer.decode(tokenized['input_ids']))
    
  # GRPO specific training arguments
    training_args = GRPOConfig(
        use_vllm=True,
        torch_empty_cache_steps=1,
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
        gradient_accumulation_steps=3,
        num_generations=16,
        max_prompt_length=2048,
        max_completion_length=2048,
        num_train_epochs=1,
        save_steps=250,
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
