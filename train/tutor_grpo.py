

import os
import sys
import asyncio
from typing import List, Tuple
from datetime import datetime
from datasets import load_dataset, load_from_disk
from unsloth import (
    FastLanguageModel, PatchFastRL, is_bfloat16_supported, get_chat_template
)
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from tutor_grpo_util import *

PatchFastRL("GRPO", FastLanguageModel)
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


async def _validate_completions(problem: str, partial_solution: str, correct_answer: str, num_attempts: int = config.completion_attempts) -> Tuple[int, int]:
    """Try completions in parallel until finding a successful one.
    Note: Completions are handled by a separate GPU service, so no memory management needed here."""
    completion_agent = CompletionAgent(port=config.completion_port)
    
    async def try_completion():
        try:
            completion = await completion_agent.generate(problem, partial_solution)
            complete_solution = partial_solution + completion
            
            model_answer = extract_answer_from_solution(complete_solution)
            if model_answer is None:
                return False
                
            numeric_answer, _ = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(correct_answer)
            
            if numeric_answer is None or correct_numeric is None:
                return False
                
            return abs(numeric_answer - correct_numeric) <= 1e-6
            
        except Exception:
            return False
    
    # Run all completion attempts in parallel
    results = await asyncio.gather(*[try_completion() for _ in range(num_attempts)])
    successful = sum(1 for r in results if r)
    return successful, len(results)

async def _validate_whole_approach_is_wrong(problem: str, solution: str, correct_answer: str) -> bool:
    """Validate that the analysis section alone can lead to correct completions"""
    # Split solution into steps and get the analysis part
    steps = split_into_steps(solution)
    if not steps:
        return False
        
    # First part before steps is the analysis
    analysis = steps[0]
    
    # Try completions starting with just the analysis
    successful, total = await _validate_completions(
        problem,
        analysis,
        correct_answer,
        config.completion_attempts
    )
    
    return successful == 0 and total == config.completion_attempts

async def _validate_step_identification(
    problem: str,
    steps: List[str],
    step_num: int,
    substitution: str,
    correct_answer: str
) -> Tuple[bool, float]:
    """Validate step identification and correction in parallel.
    Returns (is_valid, improvement_bonus)"""
    # Run both validations in parallel
    wrong_partial = "".join(steps[:step_num])
    corrected_partial = "".join(steps[:step_num-1]) + substitution
    
    wrong_check, fixed_check = await asyncio.gather(
        _validate_completions(problem, wrong_partial, correct_answer, config.completion_attempts),
        _validate_completions(problem, corrected_partial, correct_answer, config.completion_attempts)
    )
    
    successful_wrong, total_wrong = wrong_check
    successful_fixed, total_fixed = fixed_check
    
    # Calculate improvement bonus based on relative success rate
    improvement_bonus = 0.0
    if successful_wrong == 0:  # Only reward if original step had no successful completions
        success_rate = successful_fixed / total_fixed
        if 0.1 < success_rate <= 0.4:  # 10-40%
            improvement_bonus = 0.1
        elif 0.4 < success_rate <= 0.7:  # 40-70%
            improvement_bonus = 0.2
        elif success_rate > 0.7:         # >70%
            improvement_bonus = 0.3
    
    is_valid = successful_wrong == 0 and successful_fixed > 0
    return is_valid, improvement_bonus

def main():
    # Setup logging
    logger = setup_training_logger(config.model_type)
    stats = ValidationStats()
    
    # Setup callback for logging
    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logger.info(f"\nValidation Statistics:\n{stats.get_summary()}")
    
    def simple_reward_func(completions, **kwargs) -> list[float]:
        """Simple reward function that checks for basic structure"""
        rewards = []
        
        for completion in completions:
            reward = 0.0
            
            # Basic structure check
            if "Analysis:" in completion:
                reward += 0.3
                
            if "Verdict:" in completion:
                reward += 0.3
                
            if "Correction:" in completion:
                reward += 0.4
                
            rewards.append(reward)
        return rewards

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=4096,
        fast_inference=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
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
        loftq_config=None)

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
        map_eos_token=True)

    # Load dataset and format it
    dataset =  load_from_disk(config.dataset_name)
    
    def formatting_func(example):
        # Wrap the prompt in INST tags while keeping all fields
        return {
            **example,  # Keep all original fields
            "prompt": f"[INST]{example['prompt']}[/INST]"
        }

    # Apply formatting to dataset
    formatted_dataset = dataset.map(
        formatting_func,
        desc="Adding INST tags to prompts"
    )

    # Create timestamped output directory with model_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{config.model_type}/{timestamp}"

    # GRPO specific training arguments
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        torch_empty_cache_steps=10,
        learning_rate = 3e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.05,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 3,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations =5, # Decrease if out of memory
        max_prompt_length = 1348,
        max_completion_length = 5148,
        num_train_epochs = 1, # Set to 1 for a full training run
        save_steps = 250, 
        max_grad_norm = 0.1,
        report_to = "none", # Can use Weights & Biases
        output_dir = output_dir,
    )


    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[simple_reward_func],
        args=training_args,
        train_dataset=formatted_dataset
    )

    # Train the model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    # Save both merged model and LoRA weights
    logger.info("Saving model...")
    models_dir = "models"
    os.makedirs(os.path.join(models_dir, config.model_type), exist_ok=True)
    model_output_dir = os.path.join(models_dir, config.model_type, timestamp)
    
    # Save the merged model
    try:
        model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
        logger.info(f"Merged model saved to {model_output_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

if __name__ == "__main__":
    main()
