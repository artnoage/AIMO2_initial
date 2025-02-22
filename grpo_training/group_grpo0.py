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
from rewards import GroupReward, SolutionSimilarityChecker

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
    def __init__(self, reward_func, logger, save_frequency=100):
        self.reward_func = reward_func
        self.save_frequency = save_frequency
        self.step = 0
        self.logger = logger
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        
        if logs and 'rewards/0' in logs and hasattr(self.reward_func, 'stats'):
            # Key group performance metrics for wandb
            wandb_stats = {
                'group_reward': logs['rewards/0'],
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0),
                'solution_diversity': self.reward_func.stats.group_stats.get('solution_diversity', 0.0),
                'unanimous_correct_ratio': self.reward_func.stats.group_stats.get('unanimous_correct', 0) / max(1, self.reward_func.stats.total_batches)
            }
            
            # Detailed stats for local logging only
            local_stats = {
                'reward_components': {
                    'base_rewards': self.reward_func.stats.reward_components.get('base_rewards', 0) - getattr(self, '_last_base_rewards', 0),
                    'majority_bonuses': self.reward_func.stats.reward_components.get('majority_bonuses', 0) - getattr(self, '_last_majority_bonuses', 0),
                    'diversity_bonuses': self.reward_func.stats.reward_components.get('diversity_bonuses', 0) - getattr(self, '_last_diversity_bonuses', 0),
                    'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0) - getattr(self, '_last_length_penalties', 0.0)
                },
                'group_stats': {
                    'correct_answers': self.reward_func.stats.group_stats.get('correct_answers', 0) - getattr(self, '_last_correct_answers', 0),
                    'incorrect_answers': self.reward_func.stats.group_stats.get('incorrect_answers', 0) - getattr(self, '_last_incorrect_answers', 0),
                    'unique_solutions': self.reward_func.stats.group_stats.get('unique_solutions', 0) - getattr(self, '_last_unique_solutions', 0),
                    'similar_solutions': self.reward_func.stats.group_stats.get('similar_solutions', 0) - getattr(self, '_last_similar_solutions', 0),
                    'majority_votes': self.reward_func.stats.group_stats.get('majority_votes', 0) - getattr(self, '_last_majority_votes', 0),
                    'minority_votes': self.reward_func.stats.group_stats.get('minority_votes', 0) - getattr(self, '_last_minority_votes', 0),
                    'unanimous_correct': self.reward_func.stats.group_stats.get('unanimous_correct', 0) - getattr(self, '_last_unanimous_correct', 0),
                    'unanimous_incorrect': self.reward_func.stats.group_stats.get('unanimous_incorrect', 0) - getattr(self, '_last_unanimous_incorrect', 0),
                    'split_votes': self.reward_func.stats.group_stats.get('split_votes', 0) - getattr(self, '_last_split_votes', 0),
                    'average_majority_size': self.reward_func.stats.group_stats.get('average_majority_size', 0.0),
                    'average_vote_margin': self.reward_func.stats.group_stats.get('average_vote_margin', 0.0)
                }
            }
            
            # Add similarity matrix to local stats if available
            if hasattr(self.reward_func, 'similarity_checker'):
                local_stats['similarity_matrix'] = self.reward_func.similarity_checker.compute_similarity_matrix(logs.get('completions', [])).tolist()
            
            # Store current values for next round
            self._last_base_rewards = self.reward_func.stats.reward_components.get('base_rewards', 0)
            self._last_majority_bonuses = self.reward_func.stats.reward_components.get('majority_bonuses', 0)
            self._last_diversity_bonuses = self.reward_func.stats.reward_components.get('diversity_bonuses', 0)
            self._last_length_penalties = self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
            
            # Store group stats
            self._last_correct_answers = self.reward_func.stats.group_stats.get('correct_answers', 0)
            self._last_incorrect_answers = self.reward_func.stats.group_stats.get('incorrect_answers', 0)
            self._last_unique_solutions = self.reward_func.stats.group_stats.get('unique_solutions', 0)
            self._last_similar_solutions = self.reward_func.stats.group_stats.get('similar_solutions', 0)
            self._last_majority_votes = self.reward_func.stats.group_stats.get('majority_votes', 0)
            self._last_minority_votes = self.reward_func.stats.group_stats.get('minority_votes', 0)
            self._last_unanimous_correct = self.reward_func.stats.group_stats.get('unanimous_correct', 0)
            self._last_unanimous_incorrect = self.reward_func.stats.group_stats.get('unanimous_incorrect', 0)
            self._last_split_votes = self.reward_func.stats.group_stats.get('split_votes', 0)
            
            # Update logs with our metrics
            logs.update(wandb_stats)

def main():
    # Configuration
    model_type = "group_0"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name = "Metaskepsis/Numina_medium_filtered"
    
    # Setup logging first
    logger = setup_logging(model_type)
    


    # Initialize config with modified bonus values
    reward_config = RewardConfig(model_type=model_type)
    reward_config.group_majority_bonus = 0.1  # Increased from 0.2
    reward_config.group_diversity_bonus = 2  # Increased from 1.0
    
    # Setup
    logger = setup_logging(reward_config.model_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"{model_type}, MB: {reward_config.group_majority_bonus}, DB={reward_config.group_diversity_bonus}, {model_name}, {dataset_name}, {timestamp}"
    # Initialize wandb
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": reward_config.model_type,
            "dataset": dataset_name,
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
    logger.info("\nInitialized GroupReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    if hasattr(reward_func, 'stats'):
        logger.info(f"Stats total_batches: {reward_func.stats.total_batches}")
        logger.info(f"Stats reward_components: {reward_func.stats.reward_components}")
    
    # Load model
   
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # Use the model_name variable defined at the start
        max_seq_length=2800,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
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
    
        
    def get_questions(split = "train") -> Dataset:
        data = load_dataset(dataset_name)[split] # type: ignore
        data = data.map(lambda x: { # type: ignore
            'prompt': '<|im_start|>system\n' + SYSTEM_PROMPT + '<|im_end|>\n<|im_start|>user\n' + x['problem'] + '<|im_end|>\n<|im_start|>assistant\n',
            'answer': x['answer']
        }) # type: ignore
        return data # type: ignore

    formatted_dataset = get_questions()
    formatted_dataset = formatted_dataset.shuffle(seed=42)
    formatted_dataset = formatted_dataset.select(range(2000))
    formatted_dataset = formatted_dataset.select(range(1000))
    
   
    
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Answer: {entry.get('answer')}")
        print(f"Correct answer: {entry.get('correct_answer')}")
    

    # GRPO specific training arguments
    training_args = GRPOConfig(
        use_vllm=False,
        torch_empty_cache_steps=1,
        learning_rate=4e-6,
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
        num_generations=14,
        max_prompt_length=800,
        max_completion_length=2000,
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
