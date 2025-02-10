import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer
import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
from typing import Tuple, Optional, List

model_type = "tutor"
model_name = "/Home/stat/laschos/AIMO2_initial/models/tutor/20250206_212611"
dataset_name = "Metaskepsis/tutor_prompts"

# Check if model_type is in paths
if model_type not in model_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in model_name path!")
    print("!"*80 + "\n")

if model_type not in dataset_name:
    print("\n" + "!"*80)
    print(f"WARNING: model_type '{model_type}' not found in dataset_name path!")
    print("!"*80 + "\n")


def extract_sections(response: str) -> tuple[str, str, str]:
    """Extract the Analysis, Verdict and Substitution sections from the response"""
    analysis_match = re.search(r'</Analysis>\s*(.*?)\s*<Analysis>', response, re.DOTALL)
    verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
    substitution_match = re.search(r'</Substitution>\s*(.*?)\s*<Substitution>', response, re.DOTALL)
    
    analysis = analysis_match.group(1).strip() if analysis_match else None
    verdict = verdict_match.group(1).strip() if verdict_match else None
    substitution = substitution_match.group(1).strip() if substitution_match else None
    
    return analysis, verdict, substitution

async def _validate_completions(problem: str, partial_solution: str, correct_answer: str, num_attempts: int = 5) -> Tuple[int, int]:
    """Try completions until finding a successful one or reaching max attempts"""
    # TODO: Implement completion agent call
    # For now just return no successes
    return 0, num_attempts

async def _validate_whole_approach_is_wrong(problem: str, solution: str, correct_answer: str) -> bool:
    """Validate that the analysis section alone can lead to correct completions"""
    # Split solution into steps and get the analysis part
    steps = solution.split('\n')
    if not steps:
        return False
        
    # First part before steps is the analysis
    analysis = steps[0]
    
    # Try completions starting with just the analysis
    successful, total = await _validate_completions(
        problem,
        analysis,
        correct_answer,
        5  # num_attempts
    )
    
    return successful == 0 and total == 5

async def _validate_step_identification(
    problem: str,
    steps: List[str],
    step_num: int,
    substitution: str,
    correct_answer: str
) -> bool:
    """Validate step identification and correction"""
    # Try completions from the wrong step - all should fail
    wrong_partial = "".join(steps[:step_num])
    successful_wrong, total_wrong = await _validate_completions(
        problem,
        wrong_partial,
        correct_answer,
        5  # num_attempts
    )
    if successful_wrong > 0:
        return False
        
    # Try completions with correction - at least one should succeed
    corrected_partial = "".join(steps[:step_num-1]) + substitution
    successful_fixed, total_fixed = await _validate_completions(
        problem,
        corrected_partial,
        correct_answer,
        5  # num_attempts
    )
    
    return successful_fixed > 0 and total_fixed == 5

def main():
    def structure_reward_func(completions, problem: str, model_solution: str, correct_answer: str, **kwargs) -> list[float]:
        """Reward function that checks if the response has valid structure"""
        rewards = []
        for completion in completions:
            # Extract sections
            analysis, verdict, substitution = extract_sections(completion)
            
            # Verdict must exist
            if verdict is None:
                rewards.append(0.0)
                continue
                
            reward = 0.0
            
            # Check if verdict is in valid format
            valid_verdicts = ["The answer is correct", "The whole approach is wrong"]
            is_step_verdict = verdict.startswith("Step ")
            if not (verdict in valid_verdicts or is_step_verdict):
                rewards.append(0.0)
                continue
            
            # Give points for valid verdict format
            reward += 0.1
            
            # Give points for analysis if present
            if analysis is not None:
                reward += 0.05
            
            # Check substitution based on verdict type
            if is_step_verdict:
                # For step verdicts, substitution must exist
                if substitution is None:
                    rewards.append(0.0)
                    continue
                reward += 0.05
            else:
                # For other verdicts, substitution must be None
                if substitution is not None:
                    rewards.append(0.0)
                    continue
                reward += 0.05
            
            rewards.append(reward)
            
        return rewards

    async def _validate_completions(problem: str, partial_solution: str, correct_answer: str, num_attempts: int = 5) -> Tuple[int, int]:
        """Try completions until finding a successful one or reaching max attempts"""
        # TODO: Implement completion agent call
        # For now just return no successes
        return 0, num_attempts

    async def _validate_whole_approach_is_wrong(problem: str, solution: str, correct_answer: str) -> bool:
        """Validate that the analysis section alone can lead to correct completions"""
        # Split solution into steps and get the analysis part
        steps = solution.split('\n')
        if not steps:
            return False
            
        # First part before steps is the analysis
        analysis = steps[0]
        
        # Try completions starting with just the analysis
        successful, total = await _validate_completions(
            problem,
            analysis,
            correct_answer,
            5  # num_attempts
        )
        
        return successful == 0 and total == 5

    async def _validate_step_identification(
        problem: str,
        steps: List[str],
        step_num: int,
        substitution: str,
        correct_answer: str
    ) -> bool:
        """Validate step identification and correction"""
        # Try completions from the wrong step - all should fail
        wrong_partial = "".join(steps[:step_num])
        successful_wrong, total_wrong = await _validate_completions(
            problem,
            wrong_partial,
            correct_answer,
            5  # num_attempts
        )
        if successful_wrong > 0:
            return False
            
        # Try completions with correction - at least one should succeed
        corrected_partial = "".join(steps[:step_num-1]) + substitution
        successful_fixed, total_fixed = await _validate_completions(
            problem,
            corrected_partial,
            correct_answer,
            5  # num_attempts
        )
        
        return successful_fixed > 0 and total_fixed == 5

    async def tutor_validation_reward_func(completions, problem: str, correct_answer: str, **kwargs) -> list[float]:
        """Reward function that validates tutor responses using completion-based validation"""
        rewards = []
        max_attempts = 5  # Number of attempts to get valid verdict
        
        for completion in completions:
            valid_response = False
            attempts = 0
            reward = 0.0
            
            while not valid_response and attempts < max_attempts:
                attempts += 1
                
                # Extract sections
                analysis, verdict, substitution = extract_sections(completion)
                if not all([analysis, verdict]):
                    break  # Invalid structure, no more attempts needed
                
                # Base reward for valid structure
                reward = 0.2
                
                try:
                    # First verify if solution is actually correct
                    # TODO: Implement solution verification
                    is_correct = False  # Placeholder
                    
                    # Check if verdict agrees with actual correctness
                    tutor_says_correct = verdict == "The answer is correct"
                    if tutor_says_correct == is_correct:
                        valid_response = True
                        
                        # Additional validation based on verdict type
                        if verdict == "The answer is correct":
                            reward += 1.8  # Up to 2.0 total
                            
                        elif verdict == "The whole approach is wrong":
                            if await _validate_whole_approach_is_wrong(problem, completion, correct_answer):
                                reward += 1.8  # Up to 2.0 total
                            else:
                                reward = 0.2  # Only structure reward
                                
                        elif verdict.startswith("Step "):
                            try:
                                step_num = int(verdict.split()[1])
                                steps = completion.split('\n')
                                
                                if await _validate_step_identification(
                                    problem, steps, step_num, substitution, correct_answer
                                ):
                                    reward += 1.8  # Up to 2.0 total
                                else:
                                    reward = 0.2  # Only structure reward
                            except (ValueError, IndexError):
                                reward = 0.2  # Only structure reward
                        else:
                            reward = 0.0  # Invalid verdict type
                    
                except Exception:
                    reward = 0.0  # Handle any validation errors
                    
            rewards.append(reward)
                
        return rewards

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=6496,
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
    dataset = load_dataset(dataset_name)

    # Create timestamped output directory with model_type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"train_results/{model_type}/{timestamp}"

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
        max_prompt_length=1348,
        max_completion_length=5148,
        num_train_epochs=1,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=output_dir,
    )

    # Initialize GRPO trainer with reward functions
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            structure_reward_func,     # Check response structure
            tutor_validation_reward_func  # Validate tutor response
        ],
        args=training_args,
        train_dataset=dataset['train'],
    )

    # Train the model
    trainer.train()

    # Save both merged model and LoRA weights
    models_dir = "models"
    os.makedirs(os.path.join(models_dir, model_type), exist_ok=True)
    model_output_dir = os.path.join(models_dir, model_type, timestamp)
    
    # Save the merged model
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {model_output_dir}")

if __name__ == "__main__":
    main()
