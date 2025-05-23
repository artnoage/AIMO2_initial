import os
import wandb
import logging
import sys
from datasets import load_from_disk, Dataset
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel) # This might need to be GRPO or GRPOTrainer depending on TRL version
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import RewardConfig # Assuming RewardConfig might store dtw_max_reward
from utils.data_preparation import prepare_solution_data
from utils.agents import FULLSOLUTION_SYSTEM_PROMPT
from utils.similarity_checker import SolutionSimilarityChecker
# Changed import for the new DTW reward function
from grpo.rewards.solution_dtw_reward import SolutionDTWReward


class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass

def setup_logging(model_type: str, script_name: str = "solution_dtw_grpo") -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Use a logger name specific to this script
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to prevent duplicate logs if script is re-run
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(
        f"{log_dir}/training_{timestamp}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', # Added levelname
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout)) # Log to stdout as well
    return logger

class LoggingCallback(TrainerCallback):
    """Callback for logging training metrics, adapted for DTW"""
    def __init__(self, reward_func, logger, save_frequency=100):
        self.reward_func = reward_func
        self.save_frequency = save_frequency
        self.step = 0
        self.logger = logger
        
        # Initialize tracking variables for reward components
        self._last_base_correctness_rewards = 0.0
        self._last_dtw_similarity_rewards = 0.0
        self._last_step_count_match_rewards = 0.0 # New component
        self._last_total_examples = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        
        if logs and 'rewards/0' in logs and hasattr(self.reward_func, 'stats'):
            # current_total_examples = getattr(self.reward_func.stats, 'total_examples', 0)
            
            plurality_stats = getattr(self.reward_func.stats, 'plurality_stats', {})
            dtw_stats = getattr(self.reward_func.stats, 'dtw_stats', {})
            step_count_stats = getattr(self.reward_func.stats, 'step_count_stats', {}) # New stats
            
            latest_batch = {}
            if hasattr(self.reward_func.stats, 'batch_results') and self.reward_func.stats.batch_results:
                latest_batch = self.reward_func.stats.batch_results[-1]
            
            # WandB logging is now handled by the reward function's _finalize_batch method.
            # This callback will focus on console logging.

            # Detailed stats for local logging only (delta from last log)
            current_base_correctness = self.reward_func.stats.reward_components.get('base_correctness_rewards', 0.0)
            current_dtw_similarity = self.reward_func.stats.reward_components.get('dtw_similarity_rewards', 0.0)
            current_step_count_match = self.reward_func.stats.reward_components.get('step_count_match_rewards', 0.0)

            local_stats_msg = (
                f"Batch Rewards (delta): "
                f"BaseCorrect: {current_base_correctness - self._last_base_correctness_rewards:.2f}, "
                f"DTWSim: {current_dtw_similarity - self._last_dtw_similarity_rewards:.2f}, "
                f"StepCountMatch: {current_step_count_match - self._last_step_count_match_rewards:.2f}"
            )
            self.logger.info(local_stats_msg)
            
            if latest_batch:
                # Use the new cumulative stats from RewardStats for console logging
                cumulative_correct_all = getattr(self.reward_func.stats, 'cumulative_individual_correct_answers', 0)
                cumulative_total_all = getattr(self.reward_func.stats, 'cumulative_individual_total_answers', 0)
                cumulative_incorrect_all = cumulative_total_all - cumulative_correct_all

                if cumulative_incorrect_all > 0:
                    cumulative_cti_ratio_all_str = f"{cumulative_correct_all / cumulative_incorrect_all:.3f}"
                elif cumulative_correct_all > 0:
                    cumulative_cti_ratio_all_str = "inf"
                else:
                    cumulative_cti_ratio_all_str = "0.000"

                if cumulative_total_all > 0:
                    cumulative_ctt_ratio_all_str = f"{cumulative_correct_all / cumulative_total_all:.6f}"
                else:
                    cumulative_ctt_ratio_all_str = "0.000000"

                # Batch-specific ratios
                batch_correct_answers = latest_batch.get('correct_answers', 0)
                batch_total_answers_in_group = latest_batch.get('total_answers', 0)
                batch_incorrect_answers = batch_total_answers_in_group - batch_correct_answers

                if batch_total_answers_in_group > 0:
                    batch_ctt_ratio = batch_correct_answers / batch_total_answers_in_group
                    batch_ctt_ratio_str = f"{batch_ctt_ratio:.6f}"
                else:
                    batch_ctt_ratio_str = "0.000000"

                if batch_incorrect_answers > 0:
                    batch_cti_ratio = batch_correct_answers / batch_incorrect_answers
                    batch_cti_ratio_str = f"{batch_cti_ratio:.3f}"
                elif batch_correct_answers > 0: # incorrect is 0, correct is > 0
                    batch_cti_ratio_str = "inf"
                else: # incorrect is 0, correct is 0
                    batch_cti_ratio_str = "0.000"

                self.logger.info(
                    f"Step {self.step}: Plurality ans: {latest_batch.get('plurality_answer')} " +
                    f"({latest_batch.get('plurality_percentage', 0.0):.2%}), " +
                    f"Correct: {latest_batch.get('plurality_correct', False)}, " +
                    f"Batch correct rate: {batch_correct_answers}/{batch_total_answers_in_group}, " +
                    f"Batch CTT Ratio: {batch_ctt_ratio_str}, " +
                    f"Batch CTI Ratio: {batch_cti_ratio_str}, " +
                    f"Cumulative CTT (All Ans): {cumulative_ctt_ratio_all_str}, " +
                    f"Cumulative CTI (All Ans): {cumulative_cti_ratio_all_str}, " +
                    f"Avg DTW Dist: {dtw_stats.get('average_dtw_distance', 0.0):.4f}, " +
                    f"Avg Step Diff: {step_count_stats.get('average_step_diff', 0.0):.2f}"
                )
            
            # Store current values for next round's delta calculation
            self._last_base_correctness_rewards = current_base_correctness
            self._last_dtw_similarity_rewards = current_dtw_similarity
            self._last_step_count_match_rewards = current_step_count_match
            
            # logs.update(wandb_stats) # Removed as WandB logging is now in BaseReward


def main():
    # Configuration
    model_type = "solution_dtw_1" # Changed model_type
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/dtw2" # Keep your base model
    dataset_name = "/Home/stat/laschos/math/AIMO2_initial/local_datasets/20250518_124125" # Your dataset
    
    logger = setup_logging(model_type, script_name=f"{model_type}_train") # Use specific logger name
    
    reward_config = RewardConfig(model_type=model_type)
    # Potentially add dtw_max_reward to RewardConfig or rely on SolutionDTWReward default
    # e.g., reward_config.dtw_max_reward = 1.0 (if not already in RewardConfig)
    # reward_config.correctness_reward = 1.0 (if not already in RewardConfig)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandb_run_name = f"{model_type}_{timestamp}" # Simplified wandb name
    
    similarity_checker = SolutionSimilarityChecker(reward_config) # Reused for step embeddings
    
    # Initialize new DTW reward function
    reward_func = SolutionDTWReward(reward_config, similarity_checker)
    logger.info("\nInitialized SolutionDTWReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    logger.info(f"Has similarity checker: {hasattr(reward_func, 'similarity_checker')}")
    logger.info(f"DTW Max Reward: {reward_func.dtw_max_reward}")
    logger.info(f"Correctness Reward Value: {reward_func.correctness_reward_value}")

    wandb.init(
        project="grpo", # Your wandb project
        name=wandb_run_name,
        config={
            "base_model_name": model_name,
            "dataset_name": dataset_name,
            "model_type": reward_config.model_type,
            # Values from RewardConfig (assuming they exist or have defaults)
            "syntax_reward": getattr(reward_config, 'syntax_reward', 0), # Example if you had it
            "execution_reward": getattr(reward_config, 'execution_reward', 0), # Example
            "numeric_tolerance": reward_config.numeric_tolerance,
            # Values from SolutionDTWReward instance
            "correctness_reward_value": reward_func.correctness_reward_value,
            "dtw_max_reward": reward_func.dtw_max_reward,
            "step_count_match_max_reward": reward_func.step_count_match_max_reward, # New config
            "answer_grouping_tolerance": reward_func.answer_grouping_tolerance,
            "tracking_plurality_metrics": True,
            "tracking_dtw_metrics": True,
            "tracking_step_count_metrics": True # New tracking flag
        }
    )
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=3000, # Adjust as needed
        fast_inference=True, # Matching original script
        load_in_4bit=False, # Or True if you need 4-bit
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=0.6, # Matching original script
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=64, # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # Recommended by Unsloth
        random_state=3407,
        # use_rslora=False, # Unsloth default
        # loftq_config=None, # Unsloth default
    )
    
    def get_questions(dataset_path: str, num_copies=1) -> Dataset: # Changed num_copies default
        data = load_from_disk(dataset_path)
        formatted_data = prepare_solution_data(data, FULLSOLUTION_SYSTEM_PROMPT)
        
        if num_copies > 1:
            logger.info(f"Original dataset size: {len(formatted_data)}")
            all_copies = [formatted_data] * num_copies # Simpler way to make copies
            concatenated_dataset = Dataset.from_dict(
                {k: sum([list(d[k]) for d in all_copies], []) for k in formatted_data.features}
            )
            logger.info(f"Concatenated dataset size: {len(concatenated_dataset)} ({num_copies}x original)")
            return concatenated_dataset
        return formatted_data
    
    # Consider a smaller num_copies for faster iteration during testing
    formatted_dataset = get_questions(dataset_name, num_copies=50) # Example: 10 copies
    formatted_dataset = formatted_dataset.shuffle(seed=42) # Consistent seed
    
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
        max_completion_length=2000,
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
    
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True) # Log traceback
        if wandb.run: wandb.finish(exit_code=1) # Ensure wandb run is closed on error
        raise
        
    logger.info("Saving final model...")
    try:
        # Ensure model_type from reward_config is used for consistency
        final_model_dir = os.path.join("models", reward_config.model_type, f"final_{timestamp}")
        os.makedirs(final_model_dir, exist_ok=True) # Create directory if it doesn't exist
        
        # Unsloth recommends saving with save_pretrained_merged for inference
        model.save_pretrained_merged(final_model_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"Merged model saved to {final_model_dir}")
        
        # Optionally save LoRA adapters separately
        # adapter_model_dir = os.path.join("models", reward_config.model_type, f"adapters_{timestamp}")
        # model.save_pretrained(adapter_model_dir, tokenizer)
        # logger.info(f"LoRA adapters saved to {adapter_model_dir}")

    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}", exc_info=True)
        if wandb.run: wandb.finish(exit_code=1)
        raise
    finally:
        if wandb.run: wandb.finish()

if __name__ == "__main__":
    main()
