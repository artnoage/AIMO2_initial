import os
import wandb
import logging
import sys
from datasets import load_dataset, Dataset
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
from utils.agents import FULLSOLUTION_SYSTEM_PROMPT_WITH_REFLECTION
from grpo.solver_ref_reward import SolverReward

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
        
        # Initialize tracking variables
        self._last_base_rewards = 0
        self._last_validation_rewards = 0
        self._last_verification_rewards = 0
        self._last_total_examples = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        print(f"LOGS: {logs}")
        if logs and 'rewards/0' in logs and hasattr(self.reward_func, 'stats'):
            # Calculate new examples in this batch
            current_total_examples = self.reward_func.stats.total_examples
            new_examples = current_total_examples - getattr(self, '_last_total_examples', 0)
            self._last_total_examples = current_total_examples
            
            # Get plurality statistics if available
            plurality_stats = {}
            if hasattr(self.reward_func.stats, 'plurality_stats'):
                plurality_stats = self.reward_func.stats.plurality_stats
            
            # Get the most recent batch result if available
            latest_batch = {}
            if hasattr(self.reward_func.stats, 'batch_results') and self.reward_func.stats.batch_results:
                latest_batch = self.reward_func.stats.batch_results[-1]
            
            # Get verification statistics if available
            verification_stats = {}
            if hasattr(self.reward_func.stats, 'verification_criteria_stats'):
                verification_stats = self.reward_func.stats.verification_criteria_stats
            
            # Key performance metrics for wandb
            wandb_stats = {
                'solution_reward': logs['rewards/0'],
                'average_reward': self.reward_func.stats.reward_components.get('average_reward', 0.0),
                'correct_answers': self.reward_func.stats.reward_components.get('correct_answers', 0),
                'incorrect_answers': self.reward_func.stats.reward_components.get('incorrect_answers', 0),
                'correct_reflections': self.reward_func.stats.reward_components.get('correct_reflections', 0),
                'incorrect_reflections': self.reward_func.stats.reward_components.get('incorrect_reflections', 0),
                'average_completion_length': self.reward_func.stats.plurality_stats.get('avg_completion_length', 0.0),
            }
            
            # Add reflection statistics to wandb
            if hasattr(self.reward_func, 'reflection_stats'):
                wandb_stats.update({
                    'reflection_accuracy': self.reward_func.reflection_stats.get('self_assessment_accuracy', 0.0),
                    'total_reflections': self.reward_func.reflection_stats.get('total_reflections', 0),
                    'correct_self_assessments': self.reward_func.reflection_stats.get('correct_self_assessments', 0),
                    'incorrect_self_assessments': self.reward_func.reflection_stats.get('incorrect_self_assessments', 0),
                    'correct_answers_assessed_correct': self.reward_func.reflection_stats.get('correct_answers_assessed_correct', 0),
                    'correct_answers_assessed_incorrect': self.reward_func.reflection_stats.get('correct_answers_assessed_incorrect', 0),
                    'incorrect_answers_assessed_correct': self.reward_func.reflection_stats.get('incorrect_answers_assessed_correct', 0),
                    'incorrect_answers_assessed_incorrect': self.reward_func.reflection_stats.get('incorrect_answers_assessed_incorrect', 0),
                })
            
            # Add plurality metrics to wandb logs
            wandb_stats.update({
                'plurality_correct_rate': plurality_stats.get('plurality_correct_rate', 0.0),
                'avg_plurality_percentage': plurality_stats.get('avg_plurality_percentage', 0.0),
                'avg_completion_length': plurality_stats.get('avg_completion_length', 0.0)
            })
            
            # Add verification metrics to wandb logs if available
            if verification_stats:
                total_verifications = verification_stats.get('total_verifications', 0)
                if total_verifications > 0:
                    wandb_stats.update({
                        'verification_detailed_rate': verification_stats.get('is_detailed_count', 0) / total_verifications,
                        'verification_correct_rate': verification_stats.get('is_correct_count', 0) / total_verifications,
                        'verification_boxed_answer_rate': verification_stats.get('boxed_answer_count', 0) / total_verifications,
                        'total_verifications': total_verifications
                    })
        
            if latest_batch:
                # Convert boolean plurality_correct to float (1.0 for True, 0.0 for False)
                plurality_correct_float = 1.0 if latest_batch.get('plurality_correct', False) else 0.0
            
                # Always include these metrics in wandb logs
                wandb_stats.update({
                    'batch_plurality_correct': plurality_correct_float,
                    'batch_plurality_percentage': latest_batch.get('plurality_percentage', 0.0),
                    'batch_total_answers': latest_batch.get('total_answers', 0),
                    'batch_correct_answers': latest_batch.get('correct_answers', 0),
                    'batch_correct_rate': latest_batch.get('correct_answers', 0) / max(latest_batch.get('total_answers', 1), 1)
                })
            
                # Add answer group metrics if available
                if hasattr(self.reward_func, 'answer_grouping_tolerance'):
                    wandb_stats.update({
                        'answer_grouping_tolerance': self.reward_func.answer_grouping_tolerance
                    })
            
            # Log to console/file
            if latest_batch:
                self.logger.info(
                    f"Step {self.step}: Plurality answer: {latest_batch.get('plurality_answer')} " +
                    f"({latest_batch.get('plurality_percentage', 0.0):.2%} of answers), " +
                    f"Correct: {latest_batch.get('plurality_correct', False)}, " +
                    f"Overall rate: {plurality_stats.get('plurality_correct_rate', 0.0):.2%}, " +
                    f"Batch correct rate: {latest_batch.get('correct_answers', 0)}/{latest_batch.get('total_answers', 0)}"
                )
                
                # Log reflection statistics
                if hasattr(self.reward_func, 'reflection_stats'):
                    reflection_stats = self.reward_func.reflection_stats
                    self.logger.info(
                        f"Reflection stats: Accuracy: {reflection_stats.get('self_assessment_accuracy', 0.0):.2%}, " +
                        f"Correct assessments: {reflection_stats.get('correct_self_assessments', 0)}/" +
                        f"{reflection_stats.get('total_reflections', 0)}, " +
                        f"Correct answers assessed correctly: {reflection_stats.get('correct_answers_assessed_correct', 0)}, " +
                        f"Incorrect answers assessed correctly: {reflection_stats.get('incorrect_answers_assessed_incorrect', 0)}"
                    )
                
                # Log verification stats if available
                if verification_stats and verification_stats.get('total_verifications', 0) > 0:
                    total_verifs = verification_stats.get('total_verifications', 0)
                    self.logger.info(
                        f"Verification stats: Detailed: {verification_stats.get('is_detailed_count', 0)}/{total_verifs} " +
                        f"({verification_stats.get('is_detailed_count', 0)/total_verifs:.2%}), " +
                        f"Correct: {verification_stats.get('is_correct_count', 0)}/{total_verifs} " +
                        f"({verification_stats.get('is_correct_count', 0)/total_verifs:.2%}), " +
                        f"Boxed answer: {verification_stats.get('boxed_answer_count', 0)}/{total_verifs} " +
                        f"({verification_stats.get('boxed_answer_count', 0)/total_verifs:.2%})"
                    )
            
            # Store current values for next round
            self._last_base_rewards = self.reward_func.stats.reward_components.get('base_rewards', 0)
            self._last_validation_rewards = self.reward_func.stats.reward_components.get('validation_rewards', 0)
            self._last_verification_rewards = self.reward_func.stats.reward_components.get('verification_rewards', 0)
            
            # Update logs with our metrics
            logs.update(wandb_stats)
            wandb.log(wandb_stats, step=state.global_step)

def main():
    # Configuration
    model_type = "self_reflect_14"
    model_name = "/Home/stat/laschos/math/AIMO2_initial/models/14X"
    dataset_name = "Metaskepsis/Numina_medium"
    
    # Setup logging first
    logger = setup_logging(model_type)
    
    # Initialize config with reward values
    reward_config = RewardConfig(model_type=model_type)
    # Reward values are already set in the config class
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{reward_config.model_type}/{timestamp}"
    wandbname = f"{model_type}, {model_name}, {dataset_name}, {timestamp}"
    
    # Initialize reward function from solver_reward2
    reward_func = SolverReward(reward_config)
    logger.info("\nInitialized SolverReward:")
    logger.info(f"Has stats object: {hasattr(reward_func, 'stats')}")
    
    # Initialize wandb
    wandb.init(
        project="grpo",
        name=wandbname,
        config={
            "model_type": reward_config.model_type,
            "dataset": dataset_name,
            "base_reward": reward_config.base_reward,
            "validation_reward": 0.2,  # Fixed value in the code
            "reflection_reward": 1.0,  # Reward for correct reflection
            "answer_grouping_tolerance": reward_func.answer_grouping_tolerance,
            "tracking_plurality_metrics": True,
            "using_similarity_checker": False
        }
    )
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=3000,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=0.77,
        max_lora_rank=32)
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    
    def get_questions(split="train") -> Dataset:
        # Load dataset
        data = load_dataset(dataset_name, split=split)
        return prepare_solution_data(data, FULLSOLUTION_SYSTEM_PROMPT_WITH_REFLECTION)
    
    formatted_dataset = get_questions()
    formatted_dataset = formatted_dataset.shuffle(seed=999)
    
    # Verify first few entries
    for i in range(min(3, len(formatted_dataset))):
        entry = formatted_dataset[i]
        print(f"\nEntry {i} verification:")
        print(f"Answer: {entry.get('answer')}")
        print(f"Correct answer: {entry.get('correct_answer')}")
        print(f"Problem: {entry.get('problem')[:100]}...")
    
    # GRPO specific training arguments
    training_args = GRPOConfig(
        torch_empty_cache_steps=1,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=6,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=800,
        max_completion_length=2200,
        num_train_epochs=1,
        save_steps=20,
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
