from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import os
import wandb
import logging
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from datetime import datetime
from unsloth import is_bfloat16_supported
import sys
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

from dotenv import load_dotenv

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from dynamic_reward import DynamicReward
from utils.data_preparation import prepare_combined_data
from utils.agents import (
    TUTOR_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    ARCHITECT_SYSTEM_PROMPT,
    DUAL_PROOF_SYSTEM_PROMPT,
    TEST_DRIVEN_PROGRAMMER_SYSTEM_PROMPT,
    FINALIZATION_SYSTEM_PROMPT
)
from utils.solution_prompt import SOLUTION_PROMPTS
from utils.programmer_prompt import PROGRAMMER_PROMPTS

load_dotenv()
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

def main():
    # Configuration
    model_type = "dynamic_0"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/W"
    dataset_name = "Metaskepsis/Olympiads_medium"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Initialize config
    reward_config = RewardConfig(model_type=model_type)
    
    # Setup
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
            "base_reward": 3.0,
            "step_continuity_reward": 0.5
        }
    )
    
    # Initialize dynamic reward function
    reward_func = DynamicReward(reward_config)
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
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=6096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing=False,
        gpu_memory_utilization=0.45,
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
        use_gradient_checkpointing=False,
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
        
    def get_questions(split="train") -> Dataset:
        """Load and format dataset with full solution, completion, programming, and wait examples
        with the following distribution:
        - 35% solution examples
        - 35% programming examples
        - 15% completion examples
        - 15% wait examples
        """
        
        
        # Load the base dataset
        data = load_dataset(dataset_name,split="train")
        data=data.shuffle(seed=141)
       
        # Define the distribution
        # You can set any value to 0 to skip generating that type of example
        distribution = {
            'solution': 0.08,
            'programming': 0.08,
            'finalization': 0,
            'tutor': 0,
            'test_programming': 0.08,
            'architect': 0.04,
            'dual_proof': 0.08,
            'test_driven_programmer': 0.08
        }
        
        # Use the prepare_combined_data function with all system prompts
        return prepare_combined_data(
            data, 
            SOLUTION_PROMPTS, 
            FINALIZATION_SYSTEM_PROMPT, 
            PROGRAMMER_PROMPTS,
            TUTOR_SYSTEM_PROMPT,
            TESTER_SYSTEM_PROMPT,
            ARCHITECT_SYSTEM_PROMPT,
            DUAL_PROOF_SYSTEM_PROMPT,
            TEST_DRIVEN_PROGRAMMER_SYSTEM_PROMPT,
            tokenizer, 
            distribution)

    # Get the formatted dataset with all types of examples
    formatted_dataset = get_questions()

   
        
    # GRPO specific training arguments
    training_args = GRPOConfig(
        torch_empty_cache_steps=1,
        learning_rate=8e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_generations=8,
        max_prompt_length=1296,
        max_completion_length=4800,
        num_train_epochs=1,
        save_steps=50,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
    )
    
    # Log the dataset structure before training
    logger.info("Dataset structure before training:")
    sample_example = formatted_dataset[0]
    for key, value in sample_example.items():
        logger.info(f"  {key}: {type(value)} - {value}")
    
    # Initialize trainer with reward function
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=formatted_dataset,
        callbacks=[LoggingCallback(reward_func=reward_func, logger=logger, save_frequency=10)]
    )
    
    # Log dataset information before training
    logger.info("Dataset information before training:")
    logger.info(f"Total examples: {len(formatted_dataset)}")
    
    # Count example types in the dataset
    example_types = {}
    for example in formatted_dataset:
        et = example.get('example_type', 'unknown')
        example_types[et] = example_types.get(et, 0) + 1
    
    logger.info(f"Example types in dataset: {example_types}")
    
    # Log a sample batch structure
    sample_batch = {
        'prompt': [formatted_dataset[i]['prompt'] for i in range(min(3, len(formatted_dataset)))],
        'answer': [formatted_dataset[i]['answer'] for i in range(min(3, len(formatted_dataset)))],
        'example_type': [formatted_dataset[i]['example_type'] for i in range(min(3, len(formatted_dataset)))]
    }
    
    logger.info("Sample batch structure:")
    for key, value in sample_batch.items():
        if key != 'prompt':  # Skip logging the full prompts
            logger.info(f"  {key}: {value}")
    
    # The example_type is already in the dataset, no need to add it again
    # Just verify that it's present in all examples
    example_type_missing = sum(1 for example in formatted_dataset if "example_type" not in example)
    if example_type_missing > 0:
        logger.warning(f"Found {example_type_missing} examples without example_type field")
    else:
        logger.info("All examples have example_type field correctly set")
    
    # Print a few examples to verify example_type is set correctly
    for i in range(min(5, len(formatted_dataset))):
        logger.info(f"Example {i} type: {formatted_dataset[i]['example_type']}")
    
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
