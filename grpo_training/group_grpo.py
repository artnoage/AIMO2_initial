import os
import wandb
import logging
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
from rewards import GroupReward, SolutionSimilarityChecker


def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('group_grpo')
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
        
        # Print statistics summary every step
        print(f"\nStep {self.step} Statistics:")
        if hasattr(self.reward_func, 'stats'):
            print("Reward function stats object exists")
            print(f"Total batches: {self.reward_func.stats.total_batches}")
            print(f"Reward components: {self.reward_func.stats.reward_components}")
            print(f"Group stats: {self.reward_func.stats.group_stats}")
            print(self.reward_func.stats.get_summary())
        else:
            print("WARNING: No stats object found in reward function")
        
        # Log to wandb
        if logs:
            # Add reward function specific metrics
            if 'rewards/0' in logs and hasattr(self.reward_func, 'stats'):
                # Calculate non-accumulative rewards for this round
                current_rewards = {
                    'group_reward': logs['rewards/0'],
                    'base_rewards': self.reward_func.stats.reward_components.get('base_rewards', 0) - getattr(self, '_last_base_rewards', 0),
                    'majority_bonuses': self.reward_func.stats.reward_components.get('majority_bonuses', 0) - getattr(self, '_last_majority_bonuses', 0),
                    'diversity_bonuses': self.reward_func.stats.reward_components.get('diversity_bonuses', 0) - getattr(self, '_last_diversity_bonuses', 0),
                    'unique_solutions': self.reward_func.stats.group_stats.get('unique_solutions', 0) - getattr(self, '_last_unique_solutions', 0),
                    'similar_solutions': self.reward_func.stats.group_stats.get('similar_solutions', 0) - getattr(self, '_last_similar_solutions', 0),
                    'avg_similarity': self.reward_func.stats.group_stats.get('total_similarity', 0.0) / max(1, self.reward_func.stats.total_batches)
                }
                
                # Store current values for next round
                self._last_base_rewards = self.reward_func.stats.reward_components.get('base_rewards', 0)
                self._last_majority_bonuses = self.reward_func.stats.reward_components.get('majority_bonuses', 0)
                self._last_diversity_bonuses = self.reward_func.stats.reward_components.get('diversity_bonuses', 0)
                self._last_unique_solutions = self.reward_func.stats.group_stats.get('unique_solutions', 0)
                self._last_similar_solutions = self.reward_func.stats.group_stats.get('similar_solutions', 0)
                
                # Log current rewards and print them
                wandb.log(current_rewards)
                print("\nCurrent Rewards:")
                for k, v in current_rewards.items():
                    print(f"{k}: {v}")
                    
            wandb.log(logs)

def main():
    # Configuration
    model_type = "group"
    model_name = "/Home/stat/laschos/AIMO2_initial/models/merged/20250209_231739"
    dataset_path = "Metaskepsis/Numina_medium"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Check if model exists locally or in HF
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
    
    # Initialize wandb
    wandb.init(
        project="group_grpo",
        name=f"group_grpo_{timestamp}",
        config={
            "model_type": reward_config.model_type,
            "dataset": dataset_path,
            "base_reward": 3.0,
            "diversity_bonus": 0.3,
            "majority_bonus": 0.2,
            "similarity_threshold_low": 0.7,
            "similarity_threshold_high": 0.9
        }
    )
    
    # Initialize similarity checker first
    similarity_checker = SolutionSimilarityChecker(reward_config)
    
    # Initialize reward function with existing config and similarity checker
    reward_func = GroupReward(reward_config, similarity_checker)
    print("\nInitialized GroupReward:")
    print(f"Has stats object: {hasattr(reward_func, 'stats')}")
    if hasattr(reward_func, 'stats'):
        print(f"Stats total_batches: {reward_func.stats.total_batches}")
        print(f"Stats reward_components: {reward_func.stats.reward_components}")
    
    # Load model
    PatchFastRL("GRPO", FastLanguageModel) 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= model_name,
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

    def formatting_func(example):
        solver_prompt = (
            "Here is a mathematical problem:\n\n"
            f"{example['problem']}\n\n"
            "Could you help me solve this from start to finish? First, let's analyze the problem, "
            "then walk through the solution step-by-step using LaTeX notation. "
            "Don't forget to put the final answer in a box using \\boxed{}"
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
    
    # Training arguments
    training_args = GRPOConfig(
        use_vllm=True,
        torch_empty_cache_steps=50,
        learning_rate=3e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.02,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=8,
        max_prompt_length=2048,
        max_completion_length=2048,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.5,
        gradient_checkpointing=True,
        report_to="wandb",
        output_dir=output_dir,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback(reward_func=reward_func, save_frequency=1)]
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
