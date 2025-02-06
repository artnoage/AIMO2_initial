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
        response = await get_model_response(self.model, prompt, max_tokens=6000)
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
        response = await get_model_response(self.model, prompt, max_tokens=6000)
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
        response = await get_model_response(self.model, prompt, max_tokens=6000)
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
        response = await get_model_response(self.model, prompt, max_tokens=6000)
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
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return (prompt[0].content, response) if return_prompt else response


class TutorAgent:
    """Agent that evaluates mathematical solutions and identifies the first wrong step"""
    
    def __init__(self, model):
        self.model = model
        
    async def find_first_wrong_step(self, problem: str, solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Analyze a solution and identify the first step that contains an error.
        Returns the step number and explanation of the error.
        """
        prompt = [
            HumanMessage(content=(
                "Here is a mathematical problem and a proposed solution:\n\n"
                f"Problem:\n{problem}\n\n"
                f"Proposed Solution:\n{solution}\n\n"
                "Please carefully read this solution step by step. "
                "If you find any errors, identify the FIRST step where something goes wrong "
                "and explain the error. If the solution is completely correct, say so.\n\n"
                "Format your response as:\n"
                "First error in Step X. <\EXPLANATION> provide explenation here <EXPLANATION>\n"
                "or\n"
                "Solution is correct."
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=6000)
        return (prompt[0].content, response) if return_prompt else response


class TournamentJudgeAgent:
    """Agent that evaluates and compares two mathematical solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def compare_solutions(self, problem: str, solution_a: str, solution_b: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Compare two solutions and determine which one is better.
        Returns 'A' or 'B' with explanation.
        """
        prompt = [
            HumanMessage(content=(
                "You are a mathematics judge. You will be presented with a problem and two proposed partial or full solutions: "
                "Solution A and Solution B. Your task is to thoroughly evaluate both solutions and determine which one "
                "demonstrates stronger reasoning and is more likely to be correct.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Solution A:\n{solution_a}\n\n"
                f"Solution B:\n{solution_b}\n\n"
                "Which solution is better, A or B?"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=6000)
        return (prompt[0].content, response) if return_prompt else response


class LokiAgent:
    """Agent that generates deliberately incorrect but convincing mathematical solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate a deliberately incorrect but convincing solution"""
        prompt = [
            HumanMessage(content=(
                "You are a math trickster tasked with creating a deliberately incorrect but convincing solution. "
                "Your goal is to write a solution that appears mathematically sound but contains subtle errors "
                "that would fool even a careful mathematician.\n\n"
                f"Problem:\n{problem}\n\n"
                "Please provide a complete solution that:\n"
                "1. Uses correct mathematical notation and LaTeX\n"
                "2. Follows logical steps\n"
                "3. Contains subtle but significant errors\n"
                "4. Arrives at an incorrect answer\n"
                "5. Appears convincing at first glance\n\n"
                "Make sure to include analysis, step-by-step reasoning, and box the final answer using \\boxed{}"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return (prompt[0].content, response) if return_prompt else response
