import os
import wandb
import logging
from datasets import load_dataset, concatenate_datasets, Dataset
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import sys
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import DynamicReward, SolutionSimilarityChecker

SYSTEM_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.\n\n
    <thinking>
    First, analyze the problem in depth and outline your approach.\n 
    This section should capture your reasoning, including any abstract thoughts or potential strategies.\n  
    Feel free to refine or correct your ideas as you work toward the solution.\n  
    </thinking>
    <response>\n
    <step>Step 1: Begin with the first calculation or operation\n
    Show your work clearly using LaTeX notation</step>\n\n
    <step>Step 2: Continue with the next logical step\n
    Each step should be numbered and self-contained</step>\n\n
    <step>Step N: In your final step, state your conclusion\n
    Put your final answer in \\boxed{}</step>\n
    </response>\n\n"""
    
# Prompt for completion tasks (with partial solution)
COMPLETION_PROMPT = """You will be given a mathematical problem with a partial solution. Your task is to continue the solution from where it left off.\n\n
    <thinking>
    First, analyze the problem and the partial solution provided.\n
    Understand what has been done so far and determine the next steps needed.\n
    Make sure your continuation maintains the same approach and style.\n
    </thinking>
    <response>\n
    {partial_solution}
    
    Continue from here, maintaining the same step numbering and format:
    
    <step>Step {next_step}: Continue with the next logical step\n
    Show your work clearly using LaTeX notation</step>\n\n
    <step>Step N: In your final step, state your conclusion\n
    Put your final answer in \\boxed{}</step>\n
    </response>\n\n"""

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('dynamic_grpo')
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
            }
            
            # Add dynamic reward specific metrics
            if 'solution_reward_uses' in self.reward_func.stats.reward_components:
                wandb_stats['solution_reward_uses'] = self.reward_func.stats.reward_components['solution_reward_uses']
            if 'completion_reward_uses' in self.reward_func.stats.reward_components:
                wandb_stats['completion_reward_uses'] = self.reward_func.stats.reward_components['completion_reward_uses']
            if 'random_selections' in self.reward_func.stats.reward_components:
                wandb_stats['random_selections'] = self.reward_func.stats.reward_components['random_selections']
            
            # Add solution-specific metrics if available
            if hasattr(self.reward_func.solution_reward, 'stats'):
                solution_stats = self.reward_func.solution_reward.stats
                if hasattr(solution_stats, 'group_stats'):
                    wandb_stats['solution_diversity'] = solution_stats.group_stats.get('solution_diversity', 0.0)
                    wandb_stats['correct_answers'] = solution_stats.group_stats.get('correct_answers', 0)
            
            # Add completion-specific metrics if available
            if hasattr(self.reward_func.completion_reward, 'stats'):
                completion_stats = self.reward_func.completion_reward.stats
                if hasattr(completion_stats, 'step_stats'):
                    wandb_stats['correct_step_numbering'] = completion_stats.step_stats.get('correct_step_numbering', 0)
                    wandb_stats['total_steps_completed'] = completion_stats.step_stats.get('total_steps_completed', 0)
            
            # Update logs with our metrics
            logs.update(wandb_stats)

def main():
    # Configuration
    model_type = "dynamic_0"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/qwen_sft/20250303_224627"
    dataset_name = "Metaskepsis/Olympiads_medium"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Initialize config
    reward_config = RewardConfig(model_type=model_type)
    reward_config.group_diversity_bonus = 2  # Increased from 1.0
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"{model_type}, DB={reward_config.group_diversity_bonus}, {model_name}, {dataset_name}, {timestamp}"
    
    # Initialize wandb
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
    
    # Initialize similarity checker first
    similarity_checker = SolutionSimilarityChecker(reward_config)
    
    # Initialize dynamic reward function
    reward_func = DynamicReward(reward_config, similarity_checker)
    logger.info("\nInitialized DynamicReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    if hasattr(reward_func, 'stats'):
        logger.info(f"Stats total_batches: {reward_func.stats.total_batches}")
        logger.info(f"Stats reward_components: {reward_func.stats.reward_components}")
    
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
        """Load and format dataset with both full solution and completion examples"""
        # Load the base dataset
        data = load_dataset(dataset_name, split=split)
        
        # Create full solution examples (50% of data)
        full_solution_data = data.map(lambda x: {
            'prompt': '<|im_start|>system\\n' + SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x['answer'],
            'partial_solution': ''  # Empty partial solution indicates full solution task
        })
        
        # Create completion examples (50% of data)
        # For completion examples, we'll simulate partial solutions by:
        # 1. Taking the first half of steps from existing solutions, or
        # 2. Creating synthetic partial solutions with placeholder steps
        
        def create_partial_solution(example):
            # For simplicity, we'll create a synthetic partial solution
            # In a real implementation, you might want to use actual solutions
            # and split them at random points
            
            # Create a basic partial solution with 1-2 steps
            import random
            num_steps = random.randint(1, 2)
            
            partial = "<step>Step 1: Let's analyze the problem.\n"
            partial += "We need to solve " + example['problem'][:50] + "...</step>\n\n"
            
            if num_steps > 1:
                partial += "<step>Step 2: Let's set up the equations.\n"
                partial += "Based on the problem, we can write...</step>\n\n"
            
            # Calculate the next step number for the completion
            next_step = num_steps + 1
            
            # Format the completion prompt with the partial solution
            formatted_prompt = '<|im_start|>system\\n' + COMPLETION_PROMPT.format(
                partial_solution=partial,
                next_step=next_step
            ) + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n'
            
            return {
                'prompt': formatted_prompt,
                'answer': example['answer'],
                'partial_solution': partial
            }
        
        completion_data = data.map(create_partial_solution)
        
        # Combine datasets (50% full solution, 50% completion)
        full_solution_data = full_solution_data.select(range(len(full_solution_data) // 2))
        completion_data = completion_data.select(range(len(completion_data) // 2))
        
        combined_data = concatenate_datasets([full_solution_data, completion_data])
        return combined_data

    # Get the formatted dataset with both types of examples
    formatted_dataset = get_questions()
    formatted_dataset = formatted_dataset.shuffle(seed=20)
    # Use a reasonable number of examples
    formatted_dataset = formatted_dataset.select(range(2000))
   
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Answer: {entry.get('answer')}")
        print(f"Partial solution: {entry.get('partial_solution')[:100]}...")
    
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=10,
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
