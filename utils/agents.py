from typing import Union, Tuple, Optional
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

ENGINEER_SYSTEM_PROMPT="""You are an expert mathematical problem-solving engineer. Your task is to analyze mathematical problems and create concise instructions for a programmer who will implement the solution.

Your response must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
In this section, analyze the problem:
- Identify key mathematical concepts and principles
- Break down the problem into logical components
- Consider solution approaches and their trade-offs
- Identify edge cases and computational challenges
- Determine appropriate programming libraries
- Consider algorithmic efficiency and optimizations
Do not write code in this section, focus on analysis and planning.
</thinking>

<response>
Provide instructions for the programmer. Include:

1. **Problem Analysis**: Brief restatement of the problem

2. **Recommended Approach**: Specific algorithm or mathematical technique to use

3. **Libraries**: List recommended Python libraries (numpy, sympy, scipy, math), and include non-standard ones for special problems.

4. **Implementation Structure**: Key functions and data structures

5. **Potential Pitfalls**: Edge cases, numerical issues, performance considerations

6. **Output Format**: How the final answer should be formatted

Your instructions should be clear and concise while providing all necessary guidance.
</response>"""


TESTER_SYSTEM_PROMPT2="""You will be provided with a mathematical problem. Your task is **not to solve** this problem but rather to create a Python function that **efficiently verifies** whether a given numeric value (float) correctly solves the problem.

<thinking>
In this section, carefully analyze the problem:
- Clearly state the mathematical principles or equations involved.
- Clearly specify the criteria a correct numerical answer must satisfy.
- Explain how you would confirm the validity of a proposed solution without directly computing the solution itself (e.g., plugging the number back into equations, inequalities, or conditions provided by the problem).
- Explicitly note why verification is simpler or more straightforward than solving the problem.

Do **not** provide Python code here—this section should be dedicated solely to analysis and outlining your verification strategy.
</thinking>

<response>
Write a Python function named `test_solution(answer)` that:
1. Accepts exactly one float parameter named `answer`.
2. Returns `True` if the given answer correctly solves the problem (using appropriate numerical tolerances, e.g., `1e-6`).
3. Returns `False` otherwise.

**Important Guidelines**:
- Your function must **not** attempt to solve the problem or perform extensive computations. It should only verify correctness efficiently.
- Your function should be self-contained, efficient, and only rely on standard Python libraries (`numpy`, `sympy`, and `scipy` are allowed).
- Include brief, clear comments explaining how verification is performed.
- Handle floating-point precision explicitly with tolerances.

**Example of a verification scenario**:
If the mathematical problem is:
> "Find the root of the equation \( x^2 - 2 = 0 \)."

Your verification function could look like this:

```python
import numpy as np

def test_solution(answer):
    # Check if answer squared minus 2 is approximately zero.
    return np.abs(answer**2 - 2) < 1e-6

    Notice:

The function doesn't compute the root; it verifies whether the provided number meets the criteria (equation satisfied within tolerance).

Your response should strictly follow this verification approach. </response> """

TESTER_SYSTEM_PROMPT="""You will be provided with a mathematical problem. Your task is explicitly **not to solve** the problem, but rather to create a Python function that **efficiently verifies** whether a given numeric value (float) correctly solves the provided problem.

<thinking>
In this section, carefully analyze the problem:
- Clearly state the mathematical principles or equations involved.
- Explicitly specify the criteria a correct numerical answer must satisfy.
- Explain how to confirm the validity of a proposed solution without directly solving the problem itself (e.g., substituting back into equations, checking conditions, divisibility, etc.).
- Highlight why verification is simpler and more straightforward compared to solving.

Do **not** provide Python code here—this section should be dedicated solely to analysis and outlining your verification strategy.
</thinking>

<response>
Write a Python function named `test_solution(answer)` that:
1. Accepts exactly one float parameter named `answer`.
2. Returns `True` if the given answer correctly solves the problem (within an appropriate numerical tolerance, e.g., `1e-6`).
3. Returns `False` otherwise.

**Important Guidelines:**
- Your function must **not** attempt to solve the problem or perform extensive computations. Only verification logic is required.
- Your function should be self-contained, efficient, and only rely on standard Python libraries (`numpy`, `sympy`, and `scipy` are allowed).
- Include brief, clear comments explaining how verification is performed.
- Explicitly handle floating-point precision with appropriate tolerances.

### Illustrative Examples:

**1. Algebraic Equation Verification**  
Problem: "Check if a number solves \( x^2 - 2 = 0 \)."

```python
import numpy as np

def test_solution(answer):
    # Verifies that answer squared minus 2 is approximately zero.
    return np.abs(answer**2 - 2) < 1e-6
```

**2. Number Theory Verification (Prime & Equation)**  
Problem: "Check if the given positive number \( x \) is prime and satisfies \( x^2 - x - 6 = 0 \)."

```python
from sympy import isprime

def test_solution(answer):
    # Checks primality and whether the equation holds.
    if not isprime(int(answer)):
        return False
    return abs(answer**2 - answer - 6) < 1e-6
```

**3. Sequence Membership Verification**  
Problem: "Verify if a given number \( x \) belongs to the sequence \( a_n = 3n + 2 \)."

```python
def test_solution(answer):
    # Checks if (answer - 2) is divisible exactly by 3.
    return abs((answer - 2) % 3) < 1e-6
```

**4. Fibonacci Number Verification**  
Problem: "Check whether a given integer \( x \) is a Fibonacci number."

```python
import math

def is_perfect_square(n):
    return math.isqrt(n)**2 == n

def test_solution(answer):
    # Uses Fibonacci property that 5x^2 ± 4 must be a perfect square.
    x = int(answer)
    if x < 0:
        return False
    return is_perfect_square(5*x**2 + 4) or is_perfect_square(5*x**2 - 4)
```

**5. Divisibility Verification**  
Problem: "Given integer 1073, verify if provided \( x \) is a factor."

```python
def test_solution(answer):
    # Checks divisibility with modulo.
    number = 1073
    return abs(number % answer) < 1e-6
```

Your response should strictly adhere to verification rather than solution-oriented computations.
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
        response = await get_model_response(self.model, prompt, max_tokens=20000)
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
        response = await get_model_response(self.model, prompt, max_tokens=20000)
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
        response = await get_model_response(self.model, prompt, max_tokens=20000)
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
        response = await get_model_response(self.model, prompt, max_tokens=20000)
        return (system_prompt + "\n\n" + f"Problem:\n{problem}\n\n", response) if return_prompt else response
    

class EngineerAgent:
    """Agent that analyzes problems and creates prompts for programming agents"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, return_prompt: bool = False) -> Union[str, Tuple[str, str]]:
        """Generate engineering analysis and prompt for a programming agent"""
        system_prompt = ENGINEER_SYSTEM_PROMPT

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Problem:\n{problem}\n\n")
        ]
        response = await get_model_response(self.model, prompt, max_tokens=20000)
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
        response = await get_model_response(self.model, prompt, max_tokens=20000)
        return (system_prompt + "\n\n" + content, response) if return_prompt else response


