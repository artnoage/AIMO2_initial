

import os
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datetime import datetime
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth.chat_templates import get_chat_template
from trl import GRPOConfig, GRPOTrainer
import sys
import aiohttp
import asyncio
from typing import Any, Union, Tuple
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
from typing import Optional, List
from langchain_core.messages import HumanMessage
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer


#Configuration. 
model_type = "tutor"
model_name = "/Home/stat/laschos/AIMO2_initial/models/tutor/20250206_212611"
dataset_name = "Metaskepsis/tutor_prompts"
COMPLETION_PORT = 8001
COMPLETION_ATTEMPTS = 10

# Reward configuration
STRUCTURE_BASE_REWARD = 0.1
ANALYSIS_REWARD = 0.05
SUBSTITUTION_REWARD = 0.05
SINGLE_STEP_BONUS = 0.02
MULTIPLE_STEP_PENALTY = 0.02
FULL_REWARD = 2.0

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

async def _validate_completions(problem: str, partial_solution: str, correct_answer: str, num_attempts: int = COMPLETION_ATTEMPTS) -> Tuple[int, int]:
    """Try completions in parallel until finding a successful one"""
    completion_agent = CompletionAgent(port=COMPLETION_PORT)
    
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
        COMPLETION_ATTEMPTS
    )
    
    return successful == 0 and total == COMPLETION_ATTEMPTS

async def _validate_step_identification(
    problem: str,
    steps: List[str],
    step_num: int,
    substitution: str,
    correct_answer: str
) -> bool:
    """Validate step identification and correction in parallel"""
    # Run both validations in parallel
    wrong_partial = "".join(steps[:step_num])
    corrected_partial = "".join(steps[:step_num-1]) + substitution
    
    wrong_check, fixed_check = await asyncio.gather(
        _validate_completions(problem, wrong_partial, correct_answer, COMPLETION_ATTEMPTS),
        _validate_completions(problem, corrected_partial, correct_answer, COMPLETION_ATTEMPTS)
    )
    
    successful_wrong, _ = wrong_check
    successful_fixed, _ = fixed_check
    
    return successful_wrong == 0 and successful_fixed > 0

def main():
    async def combined_reward_func(completions, problem: str, model_solution: str, correct_answer: str, **kwargs) -> list[float]:
        """Combined reward function that checks structure and validates responses"""
        rewards = []
        
        # First verify if model solution is correct
        model_answer = extract_answer_from_solution(model_solution)
        if model_answer is None:
            print("bug")
            return [0.0] * len(completions)
            
        model_numeric, _ = extract_numeric_answer(model_answer)
        correct_numeric, _ = extract_numeric_answer(correct_answer)
        
        if model_numeric is None or correct_numeric is None:
            print("bug")
            return [0.0] * len(completions)
            
        is_correct = abs(model_numeric - correct_numeric) <= 1e-6
        
        for completion in completions:
            # Extract sections
            analysis, verdict, substitution = extract_sections(completion)
            
            # Verdict must exist
            if verdict is None:
                rewards.append(0.0)
                continue
                
            # Check if verdict exists and is in valid categories
            valid_verdicts = ["The answer is correct", "The whole approach is wrong"]
            is_step_verdict = False
            reward = 0.0
            
            if verdict in valid_verdicts:
                is_step_verdict = False
                reward = STRUCTURE_BASE_REWARD
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
                reward = STRUCTURE_BASE_REWARD
            else:
                rewards.append(0.0)
                continue
            
            # Add points for analysis if present
            if analysis is not None:
                reward += ANALYSIS_REWARD
            
            # Check substitution based on verdict type
            if is_step_verdict:
                # For step verdicts, substitution must exist
                if substitution is None:
                    rewards.append(0.0)
                    continue
                    
                # Check substitution doesn't contain multiple steps
                substitution_steps = split_into_steps(substitution)
                if len(substitution_steps) > 1:
                    reward -= MULTIPLE_STEP_PENALTY
                else:
                    reward += SINGLE_STEP_BONUS
                
                # If substitution contains a boxed answer, verify it matches
                boxed_answer = extract_answer_from_solution(substitution)
                if boxed_answer:
                    numeric_value, _ = extract_numeric_answer(boxed_answer)
                    if numeric_value is not None and correct_numeric is not None:
                        if abs(numeric_value - correct_numeric) > 1e-6:
                            rewards.append(0.0)
                            continue
                    
                reward += SUBSTITUTION_REWARD
            else:
                # For other verdicts, substitution must be None
                if substitution is not None:
                    rewards.append(0.0)
                    continue
                reward += SUBSTITUTION_REWARD
                
            # If we get here, format is valid (reward = 0.2) - proceed with validation
            
            # Check if verdict agrees with actual correctness
            tutor_says_correct = verdict == "The answer is correct"
            if tutor_says_correct != is_correct:
                rewards.append(reward)  # Only format reward
                continue
                
            # Additional validation based on verdict type
            try:
                if verdict == "The answer is correct" and is_correct:
                    reward = FULL_REWARD
                    
                elif verdict == "The whole approach is wrong" and not is_correct:
                    if await _validate_whole_approach_is_wrong(problem, model_solution, correct_answer):
                        reward = FULL_REWARD
                
                elif is_step_verdict and not is_correct:
                    # Split solution into proper steps
                    solution_steps = split_into_steps(model_solution)
                    if await _validate_step_identification(
                        problem, 
                        solution_steps,
                        step_num,
                        substitution,
                        correct_answer
                    ):
                        reward = FULL_REWARD
                
            except Exception:
                pass  # Keep format reward on validation error
                
            rewards.append(reward)
                
        return rewards

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
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
