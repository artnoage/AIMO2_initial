

import os
import sys
import logging
import aiohttp
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union, Tuple, Dict, Optional
from datasets import load_dataset, load_from_disk, concatenate_datasets
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

PatchFastRL("GRPO", FastLanguageModel)
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
from typing import Optional, List
from langchain_core.messages import HumanMessage
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer


@dataclass
class TutorConfig:
    """Configuration for tutor training and validation"""
    # Model settings
    model_type: str = "tutor"
    model_name: str = "/Home/stat/laschos/AIMO2_initial/models/tutor/20250206_212611"
    dataset_name: str = "Metaskepsis/tutor_prompts"
    
    # API settings
    completion_port: int = 8001
    completion_attempts: int = 10
    
    # Reward settings
    structure_base_reward: float = 0.2
    analysis_reward: float = 0.2
    substitution_reward: float = 0.2
    single_step_bonus: float = 0.2
    multiple_step_penalty: float = 0.4
    full_reward: float = 5.0
    
    # Length penalty settings
    analysis_length_cost: float = 0.0001  # per character
    substitution_length_cost: float = 0.0005  # per character
    redundant_substitution_penalty: float = 0.1  # penalty for substitution in polar verdict
    
    # Validation settings
    numeric_tolerance: float = 1e-6

class ValidationStats:
    """Tracks validation statistics during training"""
    def __init__(self):
        self.total_batches = 0
        self.total_rewards = 0
        self.reward_distribution = {}  # Dynamic distribution based on actual rewards
        # Track section-level stats
        self.section_stats = {
            'missing_analysis': 0,
            'missing_verdict': 0,
            'missing_substitution': 0,
            'invalid_step_number': 0,
            'polar_verdict_with_substitution': 0,
            'step_verdict_without_substitution': 0,
            'multiple_steps_in_substitution': 0
        }
        self.reward_components = {
            'base_rewards': 0,
            'analysis_rewards': 0,
            'substitution_rewards': 0,
            'step_bonuses': 0,
            'step_penalties': 0,
            'total_analysis_length_penalty': 0,
            'total_substitution_length_penalty': 0,
            'improvement_bonuses': {
                '0.1': 0,  # 10-40% completions
                '0.2': 0,  # 40-70% completions
                '0.3': 0   # >70% completions
            }
        }
        self.full_reward_reasons = {
            'correct_answer': 0,
            'wrong_approach': 0,
            'step_correction': 0,
            'final_step_correct': 0
        }
        self.start_time = datetime.now()
    
    def update(self, rewards: list[float], completion: str = None):
        self.total_batches += 1
        for r in rewards:
            self.total_rewards += r
            # Round to 6 decimal places for better grouping
            r_rounded = round(r, 6)
            self.reward_distribution[r_rounded] = self.reward_distribution.get(r_rounded, 0) + 1
            
        # Track section presence and structure if completion provided
        if completion:
            analysis, verdict, substitution = extract_sections(completion)
            
            # Track basic section presence
            if analysis is None:
                self.section_stats['missing_analysis'] += 1
            elif analysis:
                self.reward_components['total_analysis_length_penalty'] += len(analysis) * config.analysis_length_cost
                
            if verdict is None:
                self.section_stats['missing_verdict'] += 1
            elif verdict.startswith("Step "):
                if substitution is None:
                    self.section_stats['step_verdict_without_substitution'] += 1
                elif split_into_steps(substitution):
                    if len(split_into_steps(substitution)) > 1:
                        self.section_stats['multiple_steps_in_substitution'] += 1
                    self.reward_components['total_substitution_length_penalty'] += len(substitution) * config.substitution_length_cost
            elif verdict in ["The answer is correct", "The whole approach is wrong"]:
                if substitution is not None:
                    self.section_stats['polar_verdict_with_substitution'] += 1
    
    def get_summary(self) -> str:
        total_samples = sum(self.reward_distribution.values())
        if total_samples == 0:
            return "No samples processed yet"
            
        elapsed = datetime.now() - self.start_time
        
        # Sort rewards for better readability
        sorted_rewards = sorted(self.reward_distribution.items())
        reward_dist_str = "\n".join(
            f"  {reward:.6f}: {count} samples" 
            for reward, count in sorted_rewards
        )
        
        basic_stats = (
            f"Training time: {elapsed}\n"
            f"Processed {self.total_batches} batches, "
            f"Average reward: {self.total_rewards/total_samples:.6f}\n"
            f"\nReward Distribution:\n{reward_dist_str}\n"
            f"\nSection Issues:\n"
            f"  Missing analysis: {self.section_stats['missing_analysis']}\n"
            f"  Missing verdict: {self.section_stats['missing_verdict']}\n"
            f"  Step verdict without substitution: {self.section_stats['step_verdict_without_substitution']}\n"
            f"  Polar verdict with substitution: {self.section_stats['polar_verdict_with_substitution']}\n"
            f"  Multiple steps in substitution: {self.section_stats['multiple_steps_in_substitution']}\n"
            f"\nLength Penalties:\n"
            f"  Total analysis length penalty: {self.reward_components['total_analysis_length_penalty']:.6f}\n"
            f"  Total substitution length penalty: {self.reward_components['total_substitution_length_penalty']:.6f}\n"
            f"\nImprovement Bonuses:\n"
            f"  10-40% completions (0.1): {self.reward_components['improvement_bonuses']['0.1']}\n"
            f"  40-70% completions (0.2): {self.reward_components['improvement_bonuses']['0.2']}\n"
            f"  >70% completions (0.3): {self.reward_components['improvement_bonuses']['0.3']}"
        )
        return basic_stats

def setup_training_logger(model_type: str) -> logging.Logger:
    """Setup logging configuration for training"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # File handler for training logs
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

# Initialize global config and stats
config = TutorConfig()

class CompletionAgent:
    """Agent that completes partial solutions using a local model"""
    
    def __init__(
        self,
        port: int = 8001,
        model: str = "default",
        temperature: float = 0,
        api_key: str = "EMPTY",
        max_retries: int = 3
    ):
        self.port = port
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_url = f"http://localhost:{port}/v1"
        
    async def _get_response(self, prompt: Any, max_tokens: Optional[int] = None) -> str:
        """Get response from model with retry logic"""
        # Convert prompt to messages format
        if hasattr(prompt, 'content'):  # LangChain message object
            messages = [{"role": "user", "content": prompt.content}]
        elif isinstance(prompt, list):  # List of messages
            messages = [{"role": "user", "content": prompt[-1].content}] if prompt else []
        else:  # String or other
            messages = [{"role": "user", "content": str(prompt)}]
            
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}"
                        }
                    ) as response:
                        if response.status != 200:
                            raise ValueError(f"Error from API: {await response.text()}")
                        
                        result = await response.json()
                        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise
                await asyncio.sleep(0.1)
                
        raise Exception(f"Failed after {self.max_retries} retries")
        
    async def generate(self, problem: str, partial_solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Complete a partial solution"""
        prompt = [
            HumanMessage(content=(
                "Here is a mathematical problem:\n\n"
                f"{problem}\n\n"
                "We've started solving it and got this far:\n\n"
                f"{partial_solution}\n\n"
                "Could you help finish this solution? Remember to put the final answer in \\boxed{}"
            ))
        ]
        response = await self._get_response(prompt, max_tokens=2048)
        return (prompt[0].content, response) if return_prompt else response


def extract_sections(response: str) -> tuple[str, str, str]:
    """Extract the Analysis, Verdict and Substitution sections from the response"""
    analysis_match = re.search(r'</Analysis>\s*(.*?)\s*<Analysis>', response, re.DOTALL)
    verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
    substitution_match = re.search(r'</Substitution>\s*(.*?)\s*<Substitution>', response, re.DOTALL)
    
    analysis = analysis_match.group(1).strip() if analysis_match else None
    verdict = verdict_match.group(1).strip() if verdict_match else None
    substitution = substitution_match.group(1).strip() if substitution_match else None
    
    return analysis, verdict, substitution

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

def split_into_steps(solution: str) -> List[str]:
    """Split solution into steps by newlines and numbering"""
    steps = []
    current_step = []
    
    for line in solution.split('\n'):
        if line.strip():  # Skip empty lines
            current_step.append(line)
            # If line starts with a number and period, it's a new step
            if re.match(r'^\d+\.', line.strip()):
                if current_step[:-1]:  # If we have previous lines
                    steps.append('\n'.join(current_step[:-1]))
                current_step = [line]
    
    if current_step:  # Add the last step
        steps.append('\n'.join(current_step))
        
    return steps

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
            elif verdict.startswith("Step "):
                # First validate step number format before accessing any steps
                try:
                    step_num = int(verdict.split()[1])
                except (ValueError, IndexError):
                    rewards.append(0.0)
                    continue
                    
                # Check step number is non-negative
                if step_num < 0:
                    rewards.append(0.0)
                    continue
                    
                # For step verdicts, substitution must exist
                if substitution is None:
                    rewards.append(0.0)
                    continue
                    
                # Now check if step number is valid for the solution
                solution_steps = split_into_steps(model_solution)
                if step_num >= len(solution_steps):
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
