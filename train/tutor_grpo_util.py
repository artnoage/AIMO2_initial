import os
import re
import logging
import aiohttp
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union
from langchain_core.messages import HumanMessage
from utils.benchmark_utils import extract_answer_from_solution, extract_numeric_answer

def extract_sections(response: str) -> tuple[str, str, str]:
    """Extract the Analysis, Verdict and Substitution sections from the response"""
    analysis_match = re.search(r'</Analysis>\s*(.*?)\s*<Analysis>', response, re.DOTALL)
    verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
    substitution_match = re.search(r'</Substitution>\s*(.*?)\s*<Substitution>', response, re.DOTALL)
    
    analysis = analysis_match.group(1).strip() if analysis_match else None
    verdict = verdict_match.group(1).strip() if verdict_match else None
    substitution = substitution_match.group(1).strip() if substitution_match else None
    
    return analysis, verdict, substitution

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

@dataclass
class TutorConfig:
    """Configuration for tutor training and validation"""
    # Model settings
    model_type: str = "tutor"
    model_name: str = "/Home/stat/laschos/AIMO2_initial/models/tutor/20250206_212611"
    dataset_name: str = "Metaskepsis/tutor_prompts"
    
    # API settings
    completion_port: int = 8004
    completion_attempts: int = 10
    
    # Reward settings
    structure_base_reward: float = 0.2
    analysis_reward: float = 0.2
    substitution_reward: float = 0.4
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
            'redundant_substitution_penalties': 0,
            'wrong_boxed_answer_penalties': 0,
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


# Initialize global config
config = TutorConfig()
