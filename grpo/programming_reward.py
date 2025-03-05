import re
import asyncio
import torch
import logging
from datetime import datetime
from pathlib import Path
import os, sys
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional, Any, Union
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.model_utils import *
from utils.solution_utils import *
from config import RewardConfig
from reward_stats import RewardStats
from rewards import BaseReward

class TimeoutException(Exception):
    """Exception raised when code execution times out"""
    pass

def extract_code_from_response(response: str) -> str:
    """Extract code from the model's response"""
    # First try to extract code from ```python blocks
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # If no code blocks, try to extract from <response> section
    response_match = re.search(r'<response>\s*(.*?)\s*</response>', response, re.DOTALL)
    if response_match:
        response_content = response_match.group(1)
        # Check if there are code blocks within the response section
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        # If no code blocks in response section, assume the entire response section is code
        return response_content
    
    # If no structured format, assume the entire response is code
    return response

def check_code_quality(code: str) -> Tuple[bool, str]:
    """Check code for syntax errors and basic linting issues"""
    # First check for syntax errors
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    
    # Check for basic issues without requiring pylint
    issues = []
    
    # Check for potentially dangerous operations
    dangerous_patterns = [
        (r'os\.system', 'Contains potentially unsafe system call'),
        (r'subprocess\.', 'Contains potentially unsafe subprocess call'),
        (r'exec\s*\(', 'Contains potentially unsafe exec call'),
        (r'eval\s*\(', 'Contains potentially unsafe eval call'),
        (r'__import__', 'Contains potentially unsafe dynamic import'),
        (r'open\s*\(.+,\s*[\'"]w', 'Contains file write operation'),
        (r'import\s+requests', 'Contains network request library'),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, code):
            issues.append(message)
    
    # If there are issues, return them
    if issues:
        return False, "Linting issues: " + "; ".join(issues)
    
    return True, "Code passed quality checks"

def run_code_safely(code: str, timeout: int = 5) -> Tuple[bool, Optional[float], str]:
    """Run the code in a safe environment with timeout and capture the output"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(code.encode('utf-8'))
    
    try:
        # Run the code with timeout
        with time_limit(timeout):
            # Use subprocess to run the code
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return False, None, f"Execution error: {result.stderr}"
            
            # Try to parse the output as a float
            output = result.stdout.strip()
            try:
                answer = float(output)
                return True, answer, "Success"
            except ValueError:
                return False, None, f"Output is not a valid number: '{output}'"
    
    except TimeoutException:
        return False, None, "Code execution timed out"
    except Exception as e:
        return False, None, f"Error running code: {str(e)}"
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass

class ProgrammingReward(BaseReward):
    """Reward class for programming solution evaluation"""
    
    __name__ = "programming_reward"
    relevant_stats = {
        'reward_components': [
            'structure_rewards', 'syntax_rewards', 'execution_rewards', 'correctness_rewards',
            'total_length_penalty', 'correct_solutions', 'syntax_valid_solutions', 
            'execution_valid_solutions', 'total_rewards', 'average_reward'
        ],
        'programming_stats': [
            'correct_solutions', 'incorrect_solutions', 'syntax_errors', 
            'execution_errors', 'timeout_errors'
        ]
    }
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        
        # Initialize programming-specific stats
        self.stats.programming_stats = {
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'syntax_errors': 0,
            'execution_errors': 0,
            'timeout_errors': 0
        }
        
    async def calculate_reward(self, completion: str, **kwargs) -> float:
        """Calculate reward for a programming solution"""
        try:
            # Get problem and correct answer
            problem = kwargs.get('problem', '')
            correct_answer = kwargs.get('answer', '')
            
            if not all([problem, correct_answer]):
                self.logger.warning("Missing required inputs for programming reward calculation")
                return 0.0
            
            # Initialize reward
            reward = 0.0
            
            # 1. Check for thinking and response sections (structure reward)
            has_thinking = bool(re.search(r'<thinking>.*?</thinking>', completion, re.DOTALL))
            has_response = bool(re.search(r'<response>.*?</response>', completion, re.DOTALL))
            
            if has_thinking and has_response:
                structure_reward = self.config.structure_reward
                reward += structure_reward
                self.stats.reward_components['structure_rewards'] = self.stats.reward_components.get('structure_rewards', 0) + 1
                self.logger.info(f"Applied structure reward: +{structure_reward:.3f}")
            else:
                self.logger.info(f"Missing {'thinking' if not has_thinking else ''} {'response' if not has_response else ''} section(s)")
            
            # Extract code from the completion
            code = extract_code_from_response(completion)
            if not code:
                self.logger.info("No code found in completion")
                return reward
            
            # 2. Check code quality (syntax reward)
            code_quality_passed, quality_message = check_code_quality(code)
            if code_quality_passed:
                syntax_reward = self.config.syntax_reward
                reward += syntax_reward
                self.stats.reward_components['syntax_rewards'] = self.stats.reward_components.get('syntax_rewards', 0) + 1
                self.stats.reward_components['syntax_valid_solutions'] = self.stats.reward_components.get('syntax_valid_solutions', 0) + 1
                self.logger.info(f"Applied syntax reward: +{syntax_reward:.3f}")
            else:
                self.logger.info(f"Code quality check failed: {quality_message}")
                self.stats.programming_stats['syntax_errors'] += 1
                return reward  # Return early if syntax is invalid
            
            # 3. Run the code and check if it produces a valid output (execution reward)
            execution_success, result, error_message = run_code_safely(code, timeout=self.config.timeout)
            if execution_success and result is not None:
                execution_reward = self.config.execution_reward
                reward += execution_reward
                self.stats.reward_components['execution_rewards'] = self.stats.reward_components.get('execution_rewards', 0) + 1
                self.stats.reward_components['execution_valid_solutions'] = self.stats.reward_components.get('execution_valid_solutions', 0) + 1
                self.logger.info(f"Applied execution reward: +{execution_reward:.3f}")
            else:
                self.logger.info(f"Code execution failed: {error_message}")
                if "timed out" in error_message:
                    self.stats.programming_stats['timeout_errors'] += 1
                else:
                    self.stats.programming_stats['execution_errors'] += 1
                return reward  # Return early if execution fails
            
            # 4. Check if the result matches the correct answer (correctness reward)
            # Convert correct_answer to float if it's not already
            try:
                if isinstance(correct_answer, str):
                    numeric_answer, _ = extract_numeric_answer(correct_answer)
                    if numeric_answer is not None:
                        correct_answer = numeric_answer
                    else:
                        correct_answer = float(correct_answer)
                else:
                    correct_answer = float(correct_answer)
            except (ValueError, TypeError):
                self.logger.info(f"Could not convert correct answer to float: {correct_answer}")
                return reward
            
            # Compare with tolerance
            is_correct = abs(correct_answer - result) <= self.config.numeric_tolerance
            if is_correct:
                correctness_reward = self.config.correctness_reward
                reward += correctness_reward
                self.stats.reward_components['correctness_rewards'] = self.stats.reward_components.get('correctness_rewards', 0) + 1
                self.stats.reward_components['correct_solutions'] = self.stats.reward_components.get('correct_solutions', 0) + 1
                self.stats.programming_stats['correct_solutions'] += 1
                self.logger.info(f"Applied correctness reward: +{correctness_reward:.3f}")
            else:
                self.stats.programming_stats['incorrect_solutions'] += 1
                self.logger.info(f"Incorrect answer: expected {correct_answer}, got {result}")
            
            # Apply length penalty
            length_penalty = len(code) * self.config.length_penalty_factor
            reward -= length_penalty
            self.stats.reward_components['total_length_penalty'] = self.stats.reward_components.get('total_length_penalty', 0.0) + length_penalty
            
            # Update total rewards and average
            self.stats.reward_components['total_rewards'] = self.stats.reward_components.get('total_rewards', 0.0) + reward
            total_samples = self.stats.total_batches
            self.stats.reward_components['average_reward'] = self.stats.reward_components.get('total_rewards', 0.0) / max(1, total_samples)
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error calculating programming reward: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0
