import logging
import asyncio
import aiohttp
from typing import Dict, List, Union, Tuple, Optional
from langchain_core.messages import HumanMessage
from utils.benchmark_utils import get_model_response


class AnalysisAgent:
    """Agent that provides problem analysis and approach"""
    
    def __init__(self, port: int = 8001, model: str = None, temperature: float = 0.7, 
                 api_key: str = "EMPTY", max_retries: int = 3, logger: Optional[logging.Logger] = None):
        self.port = port
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_retries = max_retries
        self.logger = logger if logger else logging.getLogger('completion_agent')
        self.base_url = f"http://localhost:{port}/v1"
        
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
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return (prompt[0].content, response) if return_prompt else response

class CompletionAgent:
    """Agent that completes partial solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, partial_solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Complete a partial solution"""
        system_prompt = """You will be given a mathematical problem and a partial solution. Your task is to complete the solution.

Your response MUST include both a <thinking> section and a <response> section.

<thinking>
First, analyze the problem and the partial solution carefully.
Understand what has been done so far and determine the next logical steps.
Identify the step numbering pattern and continue from there.
Make sure you understand the mathematical concepts involved.
Work through the solution mentally to ensure your approach is correct.
</thinking>

<response>
Continue the solution from where it left off, maintaining the same step numbering and style.
The partial solution will only contain the beginning of the response section with some steps.
You must continue with the next step number in sequence.

IMPORTANT: Each step must be properly enclosed in <step> and </step> tags.

For example, if the partial solution ends with Step 2, you should start with:

<step>Step 3: [Description of the step]
[Mathematical work for this step]
</step>

Continue with additional steps as needed:

<step>Step 4: [Description of the step]
[Mathematical work for this step]
</step>

In your final step, include your answer in a LaTeX boxed environment:
\\boxed{your final answer}

Make sure all your steps follow logically from the partial solution and that each step has both opening and closing tags.
</response>"""

        prompt = [
            HumanMessage(content=(
                f"{system_prompt}\n\n"
                f"Problem: {problem}\n\n"
                f"Partial Solution: {partial_solution}"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=2048)
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
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return (prompt[0].content, response) if return_prompt else response


class FullSolutionAgent:
    """Agent that provides complete solutions with analysis and steps"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate a complete solution with analysis and steps"""
        prompt = [
            HumanMessage(content=("You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.\n\n"
                                    "<thinking>"
                                    "First, analyze the problem in depth and outline your approach.\n" 
                                    "This section should capture your reasoning, including any abstract thoughts or potential strategies.\n " 
                                    "Feel free to refine or correct your ideas as you work toward the solution.\n  "
                                    "</thinking>"
                                    "<response>\n"
                                    "<step>Step 1: Begin with the first calculation or operation\n"
                                    "Show your work clearly using LaTeX notation</step>\n\n"
                                    "<step>Step 2: Continue with the next logical step\n"
                                    "Each step should be numbered and self-contained</step>\n\n"
                                    "<step>Step N: In your final step, state your conclusion\n"
                                    "Put your final answer in \\boxed{}</step>\n"
                                    "</response>\n\n"
                                    f"Here is the problem:\n{problem}\n\n"))]
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return (prompt[0].content, response) if return_prompt else response


class TutorAgent:
    """Agent that evaluates mathematical solutions and identifies the first wrong step"""
    
    def __init__(self, model):
        self.model = model
        
    async def find_first_wrong_step(self, problem: str, solution: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """
        Analyze a solution and identify the first step that contains an error.
        Returns analysis, verdict and suggested correction in a structured format.
        """
        prompt = [
            HumanMessage(content=(
                "Here is a mathematical problem and a proposed solution:\n\n"
                f"Problem:\n{problem}\n\n"
                f"Proposed Solution:\n{solution}\n\n"
                "Please analyze this solution and:\n"
                "1. Provide a brief analysis of the solution approach\n"
                "2. Carefully examine each step from the beginning and identify the VERY FIRST point where the logic goes wrong\n"
                "3. If there's a wrong step, suggest how to correct it\n\n"
                "Format your response exactly as:\n\n"
                "</Analysis>\n"
                "Analyze the solution approach and reasoning here\n"
                "<Analysis>\n\n"
                "</Verdict>\n"
                "Either: 'Step X' (where X is the FIRST step number where the logic becomes incorrect)\n"
                "Or: 'The whole approach is wrong' (if the approach is fundamentally flawed from the start)\n"
                "Or: 'The answer is correct' (if no errors are found)\n"
                "<Verdict>\n\n"
                "</Substitution>\n"
                "If a specific step is wrong, write 'Step X: ' followed by the correct version of that step\n"
                "Otherwise leave this section empty\n"
                "<Substitution>"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return (response, prompt[0].content) if return_prompt else response


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
        response = await get_model_response(self.model, prompt, max_tokens=8192)
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
