from typing import Union, Tuple, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from utils.model_utils import get_model_response
from utils.prompts import *


class FinalizationAgent:
    """Agent that finalizes partial solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, partial_solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Finalize a partial solution"""
        system_prompt = FINALIZATION_SYSTEM_PROMPT

        prompt = [SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Problem: {problem}\n\n"
                f"Partial Solution: {partial_solution}"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=5000)
        return (prompt[0].content, response) if return_prompt else response



class FullSolutionAgent:
    """Agent that provides complete solutions with analysis and steps"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate a complete solution with analysis and steps"""
        system_prompt = FULLSOLUTION_SYSTEM_PROMPT
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{problem}")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=32000)
        return (system_prompt + "\n\n" + problem, response) if return_prompt else response
    

class TutorAgent:
    """Agent that evaluates mathematical solutions and identifies the first wrong step"""
    
    def __init__(self, model):
        self.model = model
        
    async def find_first_wrong_step(self, problem: str, solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Analyze a solution and identify the first step that contains an error.
        Returns analysis, verdict and suggested correction in a structured format.
        """
        system_prompt = TUTOR_SYSTEM_PROMPT

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                "Here is a mathematical problem and a proposed solution:\n\n"
                f"Problem:\n{problem}\n\n"
                f"Proposed Solution:\n{solution}"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=5000)
        return (system_prompt + "\n\n" + problem + "\n\n" + solution, response) if return_prompt else response

class ProgrammingAgent:
    """Agent that generates Python code to solve mathematical problems"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate Python code that solves the mathematical problem"""
        system_prompt = PROGRAMMER_SYSTEM_PROMPT

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(f"Problem:\n{problem}\n\n")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=5000)
        return (system_prompt + "\n\n" + f"Problem:\n{problem}\n\n", response) if return_prompt else response
    
    

class ArchitectAgent:
    """Agent that analyzes problems and creates prompts for programming agents"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate engineering analysis and prompt for a programming agent"""
        system_prompt = ARCHITECT_SYSTEM_PROMPT

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Problem:\n{problem}\n\n")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=5000)
        return (system_prompt + "\n\n" + f"Problem:\n{problem}\n\n", response) if return_prompt else response


class TestingAgent:
    """Agent that creates test functions for mathematical problems"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate a test function that verifies solutions to the mathematical problem"""
        system_prompt = TESTER_SYSTEM_PROMPT

        content = f"Problem:\n{problem}\n\n"
        
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        response = await get_model_response(self.model, prompt, max_tokens=5000)
        return (system_prompt + "\n\n" + content, response) if return_prompt else response


class DualProofAgent:
    """Agent that provides both logical proof and programming solution"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate both a logical proof and a programming solution for the problem"""
        system_prompt = DUAL_PROOF_SYSTEM_PROMPT

        content = f"Problem:\n{problem}\n\n"
        
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        response = await get_model_response(self.model, prompt, max_tokens=5000)
        return (system_prompt + "\n\n" + content, response) if return_prompt else response


class TestDrivenProgrammerAgent:
    """Agent that provides both test suite and implementation for a problem"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate both a test suite and an implementation for the problem"""
        system_prompt = TEST_DRIVEN_PROGRAMMER_SYSTEM_PROMPT

        content = f"Problem:\n{problem}\n\n"
        
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        response = await get_model_response(self.model, prompt, max_tokens=5000)
        return (system_prompt + "\n\n" + content, response) if return_prompt else response


class ReflectiveSolutionAgent:
    """Agent that provides complete solutions with analysis, steps, and reflection"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate a complete solution with analysis, steps, and self-reflection"""
        system_prompt = FULLSOLUTION_SYSTEM_PROMPT_WITH_REFLECTION
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{problem}")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=6000)
        return (system_prompt + "\n\n" + problem, response) if return_prompt else response



class SolutionVerifierAgent:
    """Agent that verifies mathematical solutions and provides structured assessment"""
    
    def __init__(self, model):
        self.model = model
        
    async def verify(self, problem: str, solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Verify a mathematical solution and provide structured assessment
        
        Args:
            problem: The mathematical problem statement
            solution: The solution to verify (with boxed answer removed)
            return_prompt: Whether to return the prompt along with the response
            
        Returns:
            Full response from the model or tuple of (prompt, response)
        """
        system_prompt = SOLUTION_VERIFIER_SYSTEM_PROMPT
        
        content = (
            f"Problem:\n{problem}\n\n"
            f"Solution (with boxed answer removed):\n{solution}"
        )
        
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
        
        full_response = await get_model_response(self.model, prompt, max_tokens=5000)
        
        return (system_prompt + "\n\n" + content, full_response) if return_prompt else full_response
