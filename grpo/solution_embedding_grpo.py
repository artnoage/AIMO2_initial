import os
import wandb
import logging
import sys
from datasets import load_from_disk, Dataset
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from config import RewardConfig
from utils.data_preparation import prepare_solution_data
# Import system prompts from agents.py
from utils.agents import FULLSOLUTION_SYSTEM_PROMPT
from utils.similarity_checker import SolutionSimilarityChecker
from rewards import SolutionEmbeddingReward

class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass

def setup_logging(model_type: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('solution_embedding_grpo')
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
        
        # Initialize tracking variables
        self._last_syntax_rewards = 0
        self._last_execution_rewards = 0
        self._last_correctness_rewards = 0
        self._last_embedding_similarity_rewards = 0
        self._last_length_penalties = 0.0
        self._last_total_examples = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        
        if logs and 'rewards/0' in logs and hasattr(self.reward_func, 'stats'):
            # Calculate new examples in this batch
            current_total_examples = self.reward_func.stats.total_examples
            new_examples = current_total_examples - getattr(self, '_last_total_examples', 0)
            self._last_total_examples = current_total_examples
            
            # Get plurality statistics if available
            plurality_stats = {}
            if hasattr(self.reward_func.stats, 'plurality_stats'):
                plurality_stats = self.reward_func.stats.plurality_stats
            
            # Get embedding statistics if available
            embedding_stats = {}
            if hasattr(self.reward_func.stats, 'embedding_stats'):
                embedding_stats = self.reward_func.stats.embedding_stats
            
            # Get the most recent batch result if available
            latest_batch = {}
            if hasattr(self.reward_func.stats, 'batch_results') and self.reward_func.stats.batch_results:
                latest_batch = self.reward_func.stats.batch_results[-1]

            # WandB logging is now handled by the reward function's _finalize_batch method.
            # This callback will focus on console logging.
            
            # Detailed stats for local logging only
            local_stats = {
                'reward_components': {
                    'syntax_rewards': self.reward_func.stats.reward_components.get('syntax_rewards', 0) - getattr(self, '_last_syntax_rewards', 0),
                    'execution_rewards': self.reward_func.stats.reward_components.get('execution_rewards', 0) - getattr(self, '_last_execution_rewards', 0),
                    'correctness_rewards': self.reward_func.stats.reward_components.get('correctness_rewards', 0) - getattr(self, '_last_correctness_rewards', 0),
                    'embedding_similarity_rewards': self.reward_func.stats.reward_components.get('embedding_similarity_rewards', 0) - getattr(self, '_last_embedding_similarity_rewards', 0),
                    'length_penalties': self.reward_func.stats.reward_components.get('total_length_penalty', 0.0) - getattr(self, '_last_length_penalties', 0.0)
                }
            }
            
            # Log to console/file
            if latest_batch:
                self.logger.info(
                    f"Step {self.step}: Plurality answer: {latest_batch.get('plurality_answer')} " +
                    f"({latest_batch.get('plurality_percentage', 0.0):.2%} of answers), " +
                    f"Correct: {latest_batch.get('plurality_correct', False)}, " +
                    f"Overall rate: {plurality_stats.get('plurality_correct_rate', 0.0):.2%}, " +
                    f"Batch correct rate: {latest_batch.get('correct_answers', 0)}/{latest_batch.get('total_answers', 0)}, " +
                    f"Avg similarity: {embedding_stats.get('avg_similarity_score', 0.0):.4f}"
                )
            
            # Store current values for next round
            self._last_syntax_rewards = self.reward_func.stats.reward_components.get('syntax_rewards', 0)
            self._last_execution_rewards = self.reward_func.stats.reward_components.get('execution_rewards', 0)
            self._last_correctness_rewards = self.reward_func.stats.reward_components.get('correctness_rewards', 0)
            self._last_embedding_similarity_rewards = self.reward_func.stats.reward_components.get('embedding_similarity_rewards', 0)
            self._last_length_penalties = self.reward_func.stats.reward_components.get('total_length_penalty', 0.0)
            
            # Update logs with our metrics
            # logs.update(wandb_stats) # Removed as WandB logging is now in BaseReward


def main():
    # Configuration
    model_type = "solution_embedding_1"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/Qtp"
    dataset_name = "/Home/stat/laschos/math/AIMO2_initial/local_datasets/20250518_124125"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Initialize config with reward values
    reward_config = RewardConfig(model_type=model_type)
    # Reward values are already set in the config class
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"{model_type}, {model_name}, {dataset_name}, {timestamp}"
    
    # Initialize similarity checker
    similarity_checker = SolutionSimilarityChecker(reward_config)
    
    # Initialize reward function with similarity checker
    reward_func = SolutionEmbeddingReward(reward_config, similarity_checker)
    logger.info("\nInitialized SolutionEmbeddingReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    logger.info(f"Has similarity checker: {hasattr(reward_func, 'similarity_checker')}")
    
    # Initialize wandb
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": reward_config.model_type,
            "dataset": dataset_name,
            "syntax_reward": reward_config.syntax_reward,
            "execution_reward": reward_config.execution_reward,
            "correctness_reward": reward_config.correctness_reward,
            "answer_grouping_tolerance": reward_func.answer_grouping_tolerance,
            "high_similarity_threshold": reward_func.high_similarity_threshold,
            "embedding_similarity_max_reward": reward_func.embedding_similarity_max_reward,
            "tracking_plurality_metrics": True,
            "tracking_embedding_metrics": True
        }
    )
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4000,
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
    
    def get_questions(split="train", num_copies=10) -> Dataset:
        """
        Load dataset and make multiple copies of it.
        
        Args:
            split: Dataset split to use
            num_copies: Number of copies to make of the dataset
            
        Returns:
            Dataset with multiple copies concatenated
        """
        # Load dataset
        data = load_from_disk(dataset_name)
        
        # Prepare the dataset
        formatted_data = prepare_solution_data(data, FULLSOLUTION_SYSTEM_PROMPT)
        
        # Make multiple copies and concatenate them
        logger.info(f"Original dataset size: {len(formatted_data)}")
        
        all_copies = [formatted_data]
        for i in range(1, num_copies):
            all_copies.append(formatted_data)
            
        # Concatenate all copies
        concatenated_dataset = Dataset.from_dict(
            {k: sum([list(d[k]) for d in all_copies], []) for k in formatted_data.features}
        )
        
        logger.info(f"Concatenated dataset size: {len(concatenated_dataset)} ({num_copies}x original)")
        return concatenated_dataset
    
    # Get the dataset with multiple copies
    formatted_dataset = get_questions(num_copies=50)  # Create 3 copies of the dataset
    formatted_dataset = formatted_dataset.shuffle(seed=142)
    
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Answer: {entry.get('answer')}")
        print(f"Correct answer: {entry.get('correct_answer')}")
        print(f"Has model_solution: {bool(entry.get('model_solution'))}")
        if entry.get('model_solution'):
            print(f"Model solution length: {len(entry.get('model_solution'))}")
    
    # GRPO specific training arguments
    training_args = GRPOConfig(
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
        per_device_train_batch_size=14,
        gradient_accumulation_steps=8,
        num_generations=14,  # Fewer generations for solution tasks
        max_prompt_length=1000,
        max_completion_length=3000,
        num_train_epochs=1,
        save_steps=20,
        max_grad_norm=0.01,
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
