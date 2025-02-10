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
from typing import Any
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
from typing import Tuple, Optional, List
from langchain_core.messages import HumanMessage
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer

class CompletionAgent:
    """Agent that completes partial solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, partial_solution: str, return_prompt: bool = False) -> str:
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
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return response

async def get_model_response(model, prompt, max_tokens=None) -> str:
    """Get response from model with retry logic"""
    try:
        if max_tokens==None:
            response = await model.ainvoke(prompt)
        else:
            response = await model.ainvoke(prompt, max_tokens=max_tokens)
        return response.content
    except Exception as e:
        # Add small delay before retry to prevent overwhelming API
        await asyncio.sleep(0.1)
        raise

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

class CustomChat:
    """Chat model that makes requests using OpenAI chat format"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        temperature: float = 0,
        api_key: str = "EMPTY"
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        """Async call to chat completion endpoint"""
        max_tokens = kwargs.get("max_tokens", None)
        
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

        async with aiohttp.ClientSession() as session:
            try:
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
                    return type('Response', (), {
                        'content': result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    })()
            except Exception as e:
                print(f"Exception in CustomChat.ainvoke: {str(e)}")
                raise


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
    # Initialize completion agent with local model
    completion_model = CustomChat(
        model="default",
        temperature=0,
        api_key="EMPTY",
        base_url="http://localhost:8001/v1"
    )
    completion_agent = CompletionAgent(completion_model)
    
    successful = 0
    total = 0
    
    for _ in range(num_attempts):
        try:
            # Generate completion
            completion = await completion_agent.generate(problem, partial_solution)
            complete_solution = partial_solution + completion
            
            # Extract and validate answer
            model_answer = extract_answer_from_solution(complete_solution)
            if model_answer is None:
                total += 1
                continue
                
            numeric_answer, _ = extract_numeric_answer(model_answer)
            correct_numeric, _ = extract_numeric_answer(correct_answer)
            
            if numeric_answer is None or correct_numeric is None:
                total += 1
                continue
                
            # Compare answers
            if abs(numeric_answer - correct_numeric) <= 1e-6:
                successful = 1  # We only need one success
                total += 1
                break
                
            total += 1
            
        except Exception as e:
            total += 1
            
    return successful, total

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
            if verdict in valid_verdicts:
                is_step_verdict = False
            else:
                # Check if it's a valid step verdict (must be "Step X" where X is an integer)
                try:
                    if not verdict.startswith("Step "):
                        rewards.append(0.0)
                        continue
                    step_num = int(verdict.split()[1])
                    if step_num < 0:
                        rewards.append(0.0)
                        continue
                    is_step_verdict = True
                except (ValueError, IndexError):
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
            # First check basic format requirements
            analysis, verdict, substitution = extract_sections(completion)
            
            # Verdict must exist
            if verdict is None:
                rewards.append(0.0)
                continue
                
            # Check verdict format
            valid_verdicts = ["The answer is correct", "The whole approach is wrong"]
            is_step_verdict = False
            if verdict in valid_verdicts:
                is_step_verdict = False
            else:
                try:
                    if not verdict.startswith("Step "):
                        rewards.append(0.0)
                        continue
                    step_num = int(verdict.split()[1])
                    if step_num < 0:
                        rewards.append(0.0)
                        continue
                    is_step_verdict = True
                except (ValueError, IndexError):
                    rewards.append(0.0)
                    continue
            
            # Check substitution requirements
            if is_step_verdict:
                if substitution is None:
                    rewards.append(0.0)
                    continue
            else:
                if substitution is not None:
                    rewards.append(0.0)
                    continue
            
            # If we get here, format is valid - proceed with completion-based validation
            valid_response = False
            attempts = 0
            reward = 0.0
            
            while not valid_response and attempts < max_attempts:
                attempts += 1
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
                            reward = 2.0  # Full reward
                            
                        elif verdict == "The whole approach is wrong":
                            if await _validate_whole_approach_is_wrong(problem, completion, correct_answer):
                                reward = 2.0  # Full reward
                            
                        elif is_step_verdict:
                            if await _validate_step_identification(
                                problem, 
                                completion.split('\n'), 
                                step_num,
                                substitution,
                                correct_answer
                            ):
                                reward = 2.0  # Full reward
                    
                except Exception:
                    pass  # Continue to next attempt
                    
            rewards.append(reward)  # Will be 0.0 if no valid response found
                
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
