import os
import wandb
import logging
import re
from datasets import load_dataset, concatenate_datasets, Dataset
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import sys
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import random
# Ensure the project root is in sys.path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from rewards import DynamicReward, SolutionSimilarityChecker

# System prompt for full solution tasks
SOLVER_SYSTEM_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.\n\n
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
    
# System prompt for completion tasks
COMPLETION_SYSTEM_PROMPT = """You will be given a mathematical problem and a partial solution. Your task is to complete the solution.

Your response MUST include both a <thinking> section and a <response> section.

<thinking>
First, analyze the problem and the partial solution carefully.
Understand what has been done so far and determine the next logical steps.
Identify the step numbering pattern and continue from there.
Make sure you understand the mathematical concepts involved.
Work through the solution mentally to ensure your approach is correct.
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
                        
                        wandb_stats['solution_examples'] = solution_count
                        wandb_stats['completion_examples'] = completion_count
                except Exception as e:
                    self.logger.warning(f"Could not track example types: {str(e)}")
            
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
    dataset_name = "Metaskepsis/completion"
    
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
        
        # Create full solution examples (2/3 of data)
        full_solution_data = data.map(lambda x: {
            'prompt': '<|im_start|>system\\n' + SOLVER_SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + x['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
            'answer': x['answer'],
            'partial_solution': '',  # Empty partial solution indicates full solution task
            'example_type': 'solution'  # Add type for tracking
        })
        
        # Create completion examples (1/3 of data)
        def create_partial_solution(example):
            try:
                # If the example has a model_solution, extract steps from it
                if 'model_solution' in example and example['model_solution']:
                    # Extract response section
                    response_match = re.search(r'<response>(.*?)</response>', example['model_solution'], re.DOTALL)
                    if response_match:
                        response = response_match.group(1).strip()
                        
                        # Extract steps
                        step_pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
                        steps = step_pattern.findall(response)
                        
                        if len(steps) >= 2:  # Need at least 2 steps to create a partial solution
                            # Randomly decide how many steps to include (at least 1, leave at least 1)
                            random.seed(hash(example.get('id', 0)) % 10000)  # Deterministic but varied
                            split_point = random.randint(1, len(steps) - 1)
                            
                            # Create partial solution with the first 'split_point' steps
                            partial_steps = steps[:split_point]
                            partial_solution = '\n\n'.join([f'<step>{step}</step>' for step in partial_steps])
                            
                            # Format the completion prompt with the partial solution in the user section
                            formatted_prompt = '<|im_start|>system\\n' + COMPLETION_SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + \
                                f"Problem: {example['problem']}\n\nPartial Solution: {partial_solution}<|im_end|>\\n<|im_start|>assistant\\n"
                            
                            return {
                                'prompt': formatted_prompt,
                                'answer': example['answer'],
                                'partial_solution': partial_solution,
                                'example_type': 'completion'
                            }
                
                # If we couldn't create a valid partial solution, return it as a full solution example instead
                return {
                    'prompt': '<|im_start|>system\\n' + SOLVER_SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example['answer'],
                    'partial_solution': '',  # Empty partial solution indicates full solution task
                    'example_type': 'solution'  # This is now a solution example
                }
                    
            except Exception as e:
                logger.warning(f"Error creating partial solution: {str(e)}")
                # Return as a full solution example on error
                return {
                    'prompt': '<|im_start|>system\\n' + SOLVER_SYSTEM_PROMPT + '<|im_end|>\\n<|im_start|>user\\n' + example['problem'] + '<|im_end|>\\n<|im_start|>assistant\\n',
                    'answer': example['answer'],
                    'partial_solution': '',
                    'example_type': 'solution'
                }
        
        completion_data = data.map(create_partial_solution)
        
        # Combine datasets (2/3 full solution, 1/3 completion)
        # First, determine how many examples of each type to include
        total_examples = len(data)
        completion_count = total_examples // 3  # 1/3 of the data
        solution_count = total_examples - completion_count  # 2/3 of the data
        
        # Select the appropriate number of examples
        full_solution_data = full_solution_data.select(range(min(solution_count, len(full_solution_data))))
        completion_data = completion_data.select(range(min(completion_count, len(completion_data))))
        
        logger.info(f"Created {len(full_solution_data)} full solution examples and {len(completion_data)} completion examples")
        
        combined_data = concatenate_datasets([full_solution_data, completion_data])
        return combined_data

    # Get the formatted dataset with both types of examples
    formatted_dataset = get_questions()
    formatted_dataset = formatted_dataset.shuffle(seed=20)
    # Use a reasonable number of examples
    formatted_dataset = formatted_dataset.select(range(2000))
   
    # Verify first few entries
    solution_count = 0
    completion_count = 0
    
    for i in range(min(5, len(formatted_dataset))):
        entry = formatted_dataset[i]
        example_type = entry.get('example_type', 'unknown')
        
        if example_type == 'solution':
            solution_count += 1
        elif example_type == 'completion':
            completion_count += 1
            
        print(f"\nEntry {i} verification:")
        print(f"Type: {example_type}")
        print(f"Answer: {entry.get('answer')}")
        
        if entry.get('partial_solution'):
            partial = entry.get('partial_solution')
            print(f"Partial solution: {partial[:100]}..." + ("" if len(partial) <= 100 else f" ({len(partial)} chars)"))
            # Count steps in partial solution
            step_count = len(re.findall(r'<step>', partial))
            print(f"Steps in partial solution: {step_count}")
            
        # Check for completion indicators in prompt
        prompt = entry.get('prompt', '')
        has_continue = 'continue' in prompt.lower()
        has_next_step = 'next step' in prompt.lower()
        print(f"Prompt indicators: continue={has_continue}, next_step={has_next_step}")
    
    print(f"\nSample ratio: {solution_count} solution examples, {completion_count} completion examples")
    
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
