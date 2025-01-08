from typing import Dict, List, Union, Tuple
from langchain_core.messages import HumanMessage
from utils.benchmark_utils import get_model_response


class AnalysisAgent:
    """Agent that provides problem analysis and approach"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate analysis for a given problem"""
        prompt = [
            HumanMessage(content=(
                "Here is a mathematical problem:\n\n"
                f"{problem}\n\n"
                "Please analyze this problem - tell me about its type, "
                "what theorems and techniques would be useful, and how you'd "
                "approach solving it. Don't provide the actual solution yet.\n\n"
                "Start with '**Problem Analysis and Approach**:'"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=4096)
        return (prompt[0].content, response) if return_prompt else response

class NextStepAgent:
    """Agent that provides the next step in a solution"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, current_solution: str = "", return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate the next solution step"""
        input_text = (
            "Here is a mathematical problem:\n\n"
            f"{problem}\n\n"
        )
            
        if current_solution:
            input_text += f"Here's how far we've gotten with the solution:\n\n{current_solution}\n\n"
            input_text += "Could you help me with the next step? Please explain it using LaTeX notation."
        else:
            input_text += "Could you help me with the first step? Please explain it using LaTeX notation."
            
        prompt = [HumanMessage(content=input_text)]
        response = await get_model_response(self.model, prompt, max_tokens=4096)
        return (prompt[0].content, response) if return_prompt else response

class CompletionAgent:
    """Agent that completes partial solutions"""
    
    def __init__(self, model):
        self.model = model
        
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
        response = await get_model_response(self.model, prompt, max_tokens=4096)
        return (prompt[0].content, response) if return_prompt else response


class MissingStepAgent:
    """Agent that identifies and completes missing steps in mathematical solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def complete_missing_step(self, problem: str, incomplete_solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Identify and generate a missing intermediate step in a mathematical solution.
        
        Args:
            problem: The original math problem text
            incomplete_solution: Solution with a missing or unclear step
            return_prompt: Whether to return the prompt along with the response
            
        Returns:
            The generated missing step, or a tuple of (prompt, response) if return_prompt is True
        """
        prompt = [
            HumanMessage(content=(
                "Here is a mathematical problem:\n\n"
                f"{problem}\n\n"
                "I have a solution, but it's missing some steps in between:\n\n"
                f"{incomplete_solution}\n\n"
                "Could you help fill in just the missing step? Use LaTeX notation to explain it clearly."
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=4096)
        return (prompt[0].content, response) if return_prompt else response


class FullSolutionAgent:
    """Agent that provides complete solutions with analysis and steps"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate a complete solution with analysis and steps"""
        prompt = [
            HumanMessage(content=(
                "Here is a mathematical problem:\n\n"
                f"{problem}\n\n"
                "Could you help me solve this from start to finish? First, let's analyze the problem, "
                "then walk through the solution step-by-step using LaTeX notation. "
                "Don't forget to put the final answer in a box using \\boxed{}"
               ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=4096)
        return (prompt[0].content, response) if return_prompt else response
