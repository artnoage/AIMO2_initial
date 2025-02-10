

import os
import sys
import asyncio
from typing import List, Tuple
from datetime import datetime
from datasets import load_dataset
from unsloth import (
    FastLanguageModel, PatchFastRL, is_bfloat16_supported, get_chat_template
)
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer
from .tutor_grpo_util import (
    TutorConfig, ValidationStats, CompletionAgent, setup_training_logger,
    config, extract_sections, split_into_steps
)

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
    
    async def combined_reward_func(completions, problem: str, model_solution: str, correct_answer: str, **kwargs) -> list[float]:
        """Combined reward function that checks structure and validates responses"""
        rewards = []
        
        # First verify if model solution is correct
        model_answer = extract_answer_from_solution(model_solution)
        if model_answer is None:
            logger.warning(f"No boxed answer found in model solution: {model_solution[:100]}...")
            return [0.0] * len(completions)
            
        model_numeric, _ = extract_numeric_answer(model_answer)
        correct_numeric, _ = extract_numeric_answer(correct_answer)
        
        if model_numeric is None or correct_numeric is None:
            logger.warning(f"Could not extract numeric values - Model: {model_answer}, Correct: {correct_answer}")
            return [0.0] * len(completions)
            
        is_correct = abs(model_numeric - correct_numeric) <= 1e-6
        
        for completion in completions:
            # Extract sections
            analysis, verdict, substitution = extract_sections(completion)
            
            # Verdict must exist
            if verdict is None:
                logger.debug(f"Missing verdict section in completion: {completion[:100]}...")
                rewards.append(0.0)
                continue
                
            # Check if verdict exists and is in polar categories (yes/no verdicts)
            polar_verdicts = ["The answer is correct", "The whole approach is wrong"]
            is_step_verdict = False
            reward = 0.0
            
            if verdict in polar_verdicts:
                is_step_verdict = False
                # For polar verdicts, substitution should be None
                reward = config.structure_base_reward
                stats.reward_components['base_rewards'] += 1
                if substitution is not None:
                    reward -= config.redundant_substitution_penalty  # Apply penalty for having substitution in polar verdict
                    stats.reward_components['redundant_substitution_penalties'] += 1
            elif verdict.startswith("Step "):
                # First validate step number format before accessing any steps
                try:
                    step_num = int(verdict.split()[1])
                except (ValueError, IndexError):
                    stats.update([0.0], completion)
                    rewards.append(0.0)
                    continue
                    
                # Check step number is non-negative
                if step_num < 0:
                    stats.update([0.0], completion)
                    rewards.append(0.0)
                    continue
                    
                # For step verdicts, substitution must exist
                if substitution is None:
                    stats.update([0.0], completion)
                    rewards.append(0.0)
                    continue
                    
                # Now check if step number is valid for the solution
                solution_steps = split_into_steps(model_solution)
                if step_num >= len(solution_steps):
                    stats.update([0.0], completion)
                    rewards.append(0.0)
                    continue
                    
                is_step_verdict = True
                reward = config.structure_base_reward
                stats.reward_components['base_rewards'] += 1
            else:
                rewards.append(0.0)
                continue
            
            # Add points for analysis if present, with length penalty
            if analysis is not None:
                analysis_reward = config.analysis_reward - (len(analysis) * config.analysis_length_cost)
                reward += analysis_reward
                stats.reward_components['analysis_rewards'] += 1
            
            # Check substitution based on verdict type
            if is_step_verdict:
                # For step verdicts, substitution must exist
                if substitution is None:
                    rewards.append(0.0)
                    continue
                    
                # Check substitution doesn't contain multiple steps
                substitution_steps = split_into_steps(substitution)
                if len(substitution_steps) > 1:
                    reward -= config.multiple_step_penalty
                    stats.reward_components['step_penalties'] += 1
                else:
                    reward += config.single_step_bonus
                    stats.reward_components['step_bonuses'] += 1
                
                # If substitution contains a boxed answer, verify it matches
                boxed_answer = extract_answer_from_solution(substitution)
                if boxed_answer:
                    numeric_value, _ = extract_numeric_answer(boxed_answer)
                    if numeric_value is not None and correct_numeric is not None:
                        if abs(numeric_value - correct_numeric) <= 1e-6:
                            # Only give full reward if this is the last possible step
                            solution_steps = split_into_steps(model_solution)
                            if step_num == len(solution_steps) - 1:
                                # Verify no valid completions exist from here
                                successful, _ = await _validate_completions(
                                    problem,
                                    "".join(solution_steps[:step_num]) + substitution,
                                    correct_answer,
                                    config.completion_attempts
                                )
                                if successful == 0:  # No valid completions possible
                                    rewards.append(config.full_reward)
                                    stats.full_reward_reasons['final_step_correct'] += 1
                                    continue
                        else:
                            # Apply penalty for wrong boxed answer in substitution
                            reward -= 1.0  # Significant penalty for wrong answer
                            stats.reward_components['wrong_boxed_answer_penalties'] += 1
                            stats.update([reward], completion)
                            rewards.append(reward)
                            continue
                
                # Add substitution reward with length penalty
                substitution_reward = config.substitution_reward - (len(substitution) * config.substitution_length_cost)
                reward += substitution_reward
                stats.reward_components['substitution_rewards'] += 1
            else:
                # For polar verdicts we already checked substitution is None
                # For polar verdicts, no substitution length penalty since substitution is None
                reward += config.substitution_reward
                stats.reward_components['substitution_rewards'] += 1
                
            # If we get here, format is valid (reward = 0.2) - proceed with validation
            
            # Check if verdict agrees with actual correctness
            tutor_says_correct = verdict == "The answer is correct"
            if tutor_says_correct != is_correct:
                rewards.append(reward)  # Only format reward
                continue
                
            # Additional validation based on verdict type
            try:
                if verdict == "The answer is correct" and is_correct:
                    reward = config.full_reward
                    stats.full_reward_reasons['correct_answer'] += 1
                    
                elif verdict == "The whole approach is wrong" and not is_correct:
                    # First verify that analysis exists
                    if analysis is None:
                        rewards.append(reward)  # Only format reward
                        continue
                        
                    # Then verify that the approach is truly wrong and no valid completions exist
                    if await _validate_whole_approach_is_wrong(problem, model_solution, correct_answer):
                        # Also verify that analysis suggests a different approach
                        if not any(step in analysis.lower() for step in model_solution.lower().split('\n')):
                            reward = config.full_reward
                            stats.full_reward_reasons['wrong_approach'] += 1
                
                elif is_step_verdict and not is_correct:
                    # Split solution into proper steps
                    solution_steps = split_into_steps(model_solution)
                    # First check if the original step was actually wrong
                    original_step = solution_steps[step_num]
                    if original_step == substitution:
                        rewards.append(reward)  # Only format reward if suggesting same step
                        continue
                        
                    is_valid, improvement_bonus = await _validate_step_identification(
                        problem, 
                        solution_steps,
                        step_num,
                        substitution,
                        correct_answer
                    )
                    if is_valid:
                        reward = config.full_reward + improvement_bonus
                        stats.full_reward_reasons['step_correction'] += 1
                        if improvement_bonus > 0:
                            stats.reward_components['improvement_bonuses'][str(improvement_bonus)] += 1
                
            except Exception:
                pass  # Keep format reward on validation error
                
            # Update stats before appending reward
            stats.update([reward], completion)
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

    # Load dataset
    dataset = load_dataset(config.dataset_name)

    # Create timestamped output directory with model_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{config.model_type}/{timestamp}"

    # GRPO specific training arguments
    training_args = GRPOConfig(
        use_vllm=True,
        torch_empty_cache_steps=10,
        learning_rate=3e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=3,
        gradient_accumulation_steps=1,
        num_generations=5,
        max_prompt_length=3000,
        max_completion_length=1096,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=output_dir,
    )

    # Initialize GRPO trainer with combined reward function
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[combined_reward_func],
        args=training_args,
        train_dataset=dataset['train'],
        callbacks=[LoggingCallback()]
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
