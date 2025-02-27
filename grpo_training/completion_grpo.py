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

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import CompletionReward

SYSTEM_PROMPT = """You will be given a mathematical problem and a partial solution. Your task is to complete the solution.

<thinking>
First, analyze the problem and the partial solution carefully.
Understand what has been done so far and determine the next logical steps.
Identify the step numbering pattern and continue from there.
Make sure you understand the mathematical concepts involved.
</thinking>

<response>
Continue the solution from where it left off, maintaining the same step numbering and style.
The partial solution will only contain the beginning of the response section with some steps.
You must continue with the next step number in sequence.

IMPORTANT: Each step must be properly enclosed in <step> and </step> tags.

For example, if the partial solution ends with Step 2, you should start with:

<step>Step 3: [Description of the step]
[Mathematical work for this step]
</step>

Continue with additional steps as needed:

<step>Step 4: [Description of the step]
[Mathematical work for this step]
</step>

In your final step, include your answer in a LaTeX boxed environment:
\\boxed{your final answer}

Make sure all your steps follow logically from the partial solution and that each step has both opening and closing tags.
</response>
"""

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('completion_grpo')
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
    model_type = "completion"
    model_name = "/workspace/AIMO2_initial/models/Qwen"
    dataset_name = "/workspace/AIMO2_initial/local_datasets/20250227_074232"
    
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
    
    # Initialize reward function
    reward_func = CompletionReward(reward_config)
    logger.info("\nInitialized CompletionReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=0.5,
        max_lora_rank=128)
    
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
    
    # We don't need to prepare partial solutions as they're already in the dataset
    
    def get_questions(split="train") -> Dataset:
        """Load dataset for completion training"""
        logger.info(f"Loading dataset from: {dataset_name}")
        
        try:
            # Load dataset from path (local or HF)
            if os.path.exists(dataset_name):
                data = load_from_disk(dataset_name)
                if hasattr(data, 'keys') and split in data:
                    data = data[split]
            else:
                data = load_dataset(dataset_name)[split]
            
            # Format for training
            data = data.map(lambda x: {
                'prompt': '<|im_start|>system\n' + SYSTEM_PROMPT + '<|im_end|>\n<|im_start|>user\n' + 
                         f"Problem: {x['problem']}\n\nPartial Solution: {x['partial_solution']}<|im_end|>\n<|im_start|>assistant\n",
                'problem': x['problem'],
                'partial_solution': x['partial_solution'],
                'answer': x['answer']
            })
            
            logger.info(f"Loaded {len(data)} examples from dataset")
            return data
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    formatted_dataset = get_questions()
    formatted_dataset1 = formatted_dataset.shuffle(seed=42)
    formatted_dataset2 = formatted_dataset.shuffle(seed=12)
    formatted_dataset= concatenate_datasets([formatted_dataset1,formatted_dataset2])
    
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Problem: {entry.get('problem')[:100]}...")
        print(f"Partial solution: {entry.get('partial_solution')[:100]}...")
        print(f"Answer: {entry.get('answer')}")
    
    # GRPO specific training arguments
    training_args = GRPOConfig(
        torch_empty_cache_steps=1,
        learning_rate=4e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=5,
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
