from typing import Union, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from utils.model_utils import get_model_response

FINALIZATION_SYSTEM_PROMPT= """You will be given a mathematical problem and a partial solution. Your task is to finalize the solution.

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

FULLSOLUTION_SYSTEM_PROMPT="""You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.

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

SIMPLE_FULLSOLUTION_SYSTEM_PROMPT="""You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.
<response>
<step>Step 1: Begin with the first calculation or operation
Show your work clearly using LaTeX notation</step>

<step>Step 2: Continue with the next logical step
Each step should be numbered and self-contained</step>

<step>Step N: In your final step, state your conclusion
Put your final answer in \\boxed{}</step>
</response>
"""




TUTOR_SYSTEM_PROMPT = """You are a mathematical tutor who evaluates solutions and identifies errors.

You will be given a mathematical problem along with a proposed solution to analyze.

Your response must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
Analyze the solution carefully, reasoning through the steps to assess correctness.
- Identify any errors and determine where the logic first goes wrong.
- If an error is found, consider how it should be corrected.
Do not state your final verdict here—focus on logical analysis.
</thinking>

<response>
Explicitly provide your verdict and necessary corrections.

<verdict>
State exactly one of the following:
- 'Step X' (where X is the **first incorrect step number**)
- 'The answer is correct' (if no errors are found)
</verdict>

<finalization>
If an incorrect step was found, provide the corrected solution starting from that step:
- Format: '<step>Step X: [corrected version]</step>...<step>Final Step</step>'
- Otherwise, leave this section empty.
</finalization>
</response>"""


PROGRAMMER_SYSTEM_PROMPT="""You will be given a mathematical problem. Your task is to respond explicitly in two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
In this section, explicitly detail your thought process step-by-step:
- Carefully analyze the problem and identify the mathematical concepts involved.
- Clearly outline your reasoning and approach, breaking down the solution into logical, implementable steps.
- Consider any edge cases, numerical stability issues, or special conditions you might encounter.
- Clearly state your intended method before beginning any code implementation.
Do not provide any Python code in this section, only your reasoning and approach.
</thinking>

<response>
In this section, write a complete, self-contained Python program that solves the problem, based explicitly on the approach described in the thinking section above. Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed).
2. Include clear comments explaining each step of your approach within the code itself.
3. Print the final answer explicitly as a single numeric value (float or integer, as appropriate).
4. Gracefully handle potential errors or edge cases.
5. Be efficient and avoid excessive resource usage.

Do NOT include explanations outside code comments. Your response here must contain ONLY valid Python code and comments.

Example format:

```python
# Solution for the problem
import math

# Step 1: Parse the problem
# [brief explanation comment]
...

# Step 2: Solve using appropriate method
# [brief explanation comment]
...

# Calculate and print the final answer
result = ...
print(result)  # Just the number, no text
</response>"""



PROGRAMMER_SYSTEM_PROMPT2="""You will be given a mathematical problem. Your task is to respond explicitly in two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
In this section, explicitly detail your thought process step-by-step:
- Carefully analyze the problem and identify the mathematical concepts involved.
- Clearly outline your reasoning and approach, breaking down the solution into logical, implementable steps.
- Consider any edge cases, numerical stability issues, or special conditions you might encounter.
- If your solution involves calculations with very large numbers, try to find mathematical shortcuts, approximations, or properties to simplify them. Examples:
  - Using logarithmic properties instead of direct exponentiation.
  - Reducing factorials via cancellation in combinatorics.
  - Applying modular arithmetic to keep numbers manageable.
  - Recognizing closed-form formulas that avoid recursion.
- Clearly state your intended method before beginning any code implementation.

Do not provide any Python code in this section, only your reasoning and approach.
</thinking>

<response>
In this section, write a complete, self-contained Python program that solves the problem, based explicitly on the approach described in the thinking section above. Your code must:

1. Be syntactically correct and runnable with standard Python libraries. The following additional libraries are permitted:  
   `numpy`, `sympy`, `scipy`, `math`, `itertools`, `functools`, `collections`, `decimal`, `fractions`, and `networkx`.  
   If you need a specific library, explain why it is required in the thinking section.

2. "Additionally, you may use mpmath for arbitrary-precision arithmetic, gmpy2 for fast number-theoretic computations, 
cvxpy for convex optimization, pulp for linear programming, statsmodels for statistical modeling.     

3. Follow a structured format:
   - Define a function (`def solve_problem(...)`) that implements the solution.
   - Include an `if __name__ == "__main__":` block to execute the function.
   
4. Include clear comments explaining each step of your approach within the code itself.

5. Print the final answer explicitly as a single numeric value (float or integer, as appropriate).
   - If the result is a floating-point number, print it rounded to **six decimal places**.
   - Use integer format if the result is a whole number (`int(value) if value.is_integer() else value`).

6. Gracefully handle potential errors or edge cases, such as:
   - Division by zero.
   - Large inputs (ensure efficiency).
   - Floating-point precision issues.
   - Empty or invalid inputs.

7. Prioritize efficiency. If a brute-force approach is too slow, optimize using:
   - Vectorized computation (`numpy`) instead of loops when possible.
   - Recursion with memoization (`functools.lru_cache`) if applicable.
   - Modular arithmetic for large number computations.

Do **NOT** include explanations outside of code comments. Your response here must contain **ONLY** valid Python code and comments.

Example format:

```python
# Solution for the problem
import math

# Step 1: Parse the problem
# [brief explanation comment]
...

# Step 2: Solve using an appropriate method
# [brief explanation comment]
...

# Step 3: Print the final answer
result = ...
print(f"{result:.6f}")  # Prints result with six decimal places if float
</response>
"""



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
        response = await get_model_response(self.model, prompt, max_tokens=4096)
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
        response = await get_model_response(self.model, prompt, max_tokens=16384)
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
        response = await get_model_response(self.model, prompt, max_tokens=4096)
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
        response = await get_model_response(self.model, prompt, max_tokens=16384)
        return (system_prompt + "\n\n" + f"Problem:\n{problem}\n\n", response) if return_prompt else response


class ProgrammingAgent2:
    """Agent that generates Python code to solve mathematical problems"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate Python code that solves the mathematical problem"""
        system_prompt = PROGRAMMER_SYSTEM_PROMPT2

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(f"Problem:\n{problem}\n\n")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=16384)
        return (system_prompt + "\n\n" + f"Problem:\n{problem}\n\n", response) if return_prompt else response