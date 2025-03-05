from typing import Union, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from utils.model_utils import get_model_response
import asyncio



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

        prompt = [SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Problem: {problem}\n\n"
                f"Partial Solution: {partial_solution}"
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
        system_prompt = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.

<thinking>
First, analyze the problem in depth and outline your approach.
This section should capture your reasoning, including any abstract thoughts or potential strategies.
Feel free to refine or correct your ideas as you work toward the solution.
</thinking>
<response>
<step>Step 1: Begin with the first calculation or operation
Show your work clearly using LaTeX notation</step>

<step>Step 2: Continue with the next logical step
Each step should be numbered and self-contained</step>

<step>Step N: In your final step, state your conclusion
Put your final answer in \\boxed{}</step>
</response>
"""
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{problem}")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=24384)
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
                "<thinking>\n"
                "Analyze the solution approach and reasoning here\n"
                "</thinking>\n\n"
                "<verdict>\n"
                "Either: 'Step X' (where X is the FIRST step number where the logic becomes incorrect)\n"
                "Or: 'The whole approach is wrong' (if the approach is fundamentally flawed from the start)\n"
                "Or: 'The answer is correct' (if no errors are found)\n"
                "</verdict>\n\n"
                "<substitution>\n"
                "If a specific step is wrong, write 'Step X: ' followed by the correct version of that step\n"
                "Otherwise leave this section empty\n"
                "</substitution>"
            ))
        ]
        response = await get_model_response(self.model, prompt, max_tokens=8192)
        return (response, prompt[0].content) if return_prompt else response


class ProgrammingAgent:
    """Agent that generates Python code to solve mathematical problems"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate Python code that solves the mathematical problem"""
        system_prompt = """You will be given a mathematical problem. Your task is to write Python code that solves this problem.

<thinking>
First, analyze the problem carefully and determine the mathematical concepts involved.
Break down the problem into steps that can be implemented in code.
Consider edge cases and potential numerical issues.
Plan your approach before writing any code.
</thinking>

<response>
Write a complete, self-contained Python program that solves the problem.
Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed)
2. Include clear comments explaining your approach
3. Print the final answer as a single float value (or integer if appropriate)
4. Handle potential errors gracefully
5. Be efficient and not use excessive resources

DO NOT include explanations outside of code comments. Your response should ONLY contain valid Python code.

Example format:
```python
# Solution for the problem
import math

# Step 1: Parse the problem
# [explanation comment]
...

# Step 2: Solve using appropriate method
# [explanation comment]
...

# Calculate and print the final answer
result = ...
print(result)  # Just the number, no text
```
</response>"""

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{problem}")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=2048)
        return (system_prompt + "\n\n" + problem, response) if return_prompt else response


