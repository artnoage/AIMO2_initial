"""
Collection of programmer prompts for mathematical problem-solving.
These are variations of the programmer system prompt.
"""

from typing import List

# Original programmer prompt
PROGRAMMER_SYSTEM_PROMPT = """You will be given a mathematical problem, that you need to solve using Python code.

Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
Use this area as your creative scratchpad.
Feel free to capture your thoughts, abstractions, corrections, or ideas in any order and form you wish—without constraints. 
Use this freedom to ensure you've gathered all insights necessary to clearly and effectively provide the requested response.
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

# Variation 1: Emphasizing algorithm efficiency
EFFICIENT_ALGORITHM_PROMPT = """You will be given a mathematical problem, that you need to solve using Python code with a focus on algorithmic efficiency.

Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
In this section:
1. Analyze the computational complexity of different approaches
2. Consider time and space efficiency tradeoffs
3. Identify potential optimizations or mathematical shortcuts
4. Select the most efficient algorithm for implementation
</thinking>

<response>
In this section, write a complete, self-contained Python program that solves the problem efficiently. Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed).
2. Include clear comments explaining each step and noting complexity considerations.
3. Print the final answer explicitly as a single numeric value (float or integer, as appropriate).
4. Use efficient algorithms and data structures appropriate for the problem.
5. Avoid unnecessary computations or memory usage.

Do NOT include explanations outside code comments. Your response here must contain ONLY valid Python code and comments.

Example format:

```python
# Solution for the problem
import math

# Step 1: Parse the problem - O(1)
# [brief explanation comment]
...

# Step 2: Solve using appropriate method - O(n log n)
# [brief explanation comment with efficiency notes]
...

# Calculate and print the final answer
result = ...
print(result)  # Just the number, no text
</response>"""

# Variation 2: Numerical stability focus
NUMERICAL_STABILITY_PROMPT = """You will be given a mathematical problem, that you need to solve using Python code with special attention to numerical stability.

Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
In this section:
1. Identify potential numerical issues (overflow, underflow, precision loss)
2. Consider alternative formulations to improve stability
3. Determine appropriate tolerances and error handling
4. Plan how to validate numerical results
</thinking>

<response>
In this section, write a complete, self-contained Python program that solves the problem with robust numerical handling. Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed).
2. Include clear comments explaining each step and any numerical considerations.
3. Print the final answer explicitly as a single numeric value (float or integer, as appropriate).
4. Use numerically stable algorithms and techniques.
5. Include appropriate error handling and validation.

Do NOT include explanations outside code comments. Your response here must contain ONLY valid Python code and comments.

Example format:

```python
# Solution for the problem with numerical stability focus
import numpy as np

# Step 1: Parse the problem
# [brief explanation comment]
...

# Step 2: Solve using numerically stable method
# [explanation of stability considerations]
...

# Calculate and print the final answer with appropriate precision
result = ...
print(result)  # Just the number, no text
</response>"""

# Variation 3: Test-driven approach
TEST_DRIVEN_PROMPT = """You will be given a mathematical problem, that you need to solve using Python code with a test-driven approach.

Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
In this section:
1. Break down the problem into testable components
2. Design test cases for each component, including edge cases
3. Plan your implementation strategy based on these tests
4. Consider how to validate the final solution
</thinking>

<response>
In this section, write a complete, self-contained Python program that solves the problem using a test-driven approach. Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed).
2. Include test functions that verify correctness of your solution.
3. Implement the solution that passes all tests.
4. Print the final answer explicitly as a single numeric value (float or integer, as appropriate).
5. Include appropriate assertions and validation.

Do NOT include explanations outside code comments. Your response here must contain ONLY valid Python code and comments.

Example format:

```python
# Solution for the problem using test-driven approach
import math
import unittest

# Test cases for our solution
def test_solution():
    # Test basic case
    assert abs(solve_problem(standard_input) - expected_output) < 1e-9
    # Test edge cases
    assert abs(solve_problem(edge_case) - edge_case_output) < 1e-9
    
# Main solution function
def solve_problem(input_data):
    # Implementation that passes all tests
    ...
    return result

# Run tests and solve the actual problem
test_solution()
result = solve_problem(problem_input)
print(result)  # Just the number, no text
</response>"""

# Variation 4: Mathematical library focused
MATH_LIBRARY_PROMPT = """You will be given a mathematical problem, that you need to solve using Python code with appropriate mathematical libraries.

Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
In this section:
1. Identify which mathematical libraries are most suitable (numpy, scipy, sympy)
2. Determine specific functions or modules that apply to this problem
3. Consider how to leverage library features for cleaner or more efficient code
4. Plan your implementation using these libraries
</thinking>

<response>
In this section, write a complete, self-contained Python program that solves the problem using appropriate mathematical libraries. Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed).
2. Make effective use of mathematical library functions rather than implementing from scratch.
3. Include clear comments explaining each library function used and why.
4. Print the final answer explicitly as a single numeric value (float or integer, as appropriate).
5. Be concise and leverage library capabilities.

Do NOT include explanations outside code comments. Your response here must contain ONLY valid Python code and comments.

Example format:

```python
# Solution for the problem using mathematical libraries
import numpy as np
from scipy import optimize
import sympy as sp

# Step 1: Set up the problem using appropriate library
# [brief explanation of library choice]
...

# Step 2: Solve using library functions
# [explanation of functions used]
...

# Calculate and print the final answer
result = ...
print(result)  # Just the number, no text
</response>"""

# Variation 5: Educational implementation
EDUCATIONAL_CODE_PROMPT = """You will be given a mathematical problem, that you need to solve using Python code with an educational focus.

Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
Use this area as your creative scratchpad.
Feel free to capture your thoughts, abstractions, corrections, or ideas in any order and form you wish—without constraints. 
Use this freedom to ensure you've gathered all insights necessary to clearly and effectively provide the requested response.
</thinking>

<response>
In this section, write a complete, self-contained Python program that solves the problem with detailed educational comments. Your code must:
1. Be syntactically correct and runnable with standard Python libraries (numpy, sympy, scipy are allowed).
2. Include extensive comments explaining the mathematical concepts and programming techniques.
3. Implement the solution in a way that prioritizes clarity over brevity.
4. Print the final answer explicitly as a single numeric value (float or integer, as appropriate).
5. Include alternative approaches in comments where relevant.

Do NOT include explanations outside code comments. Your response here must contain ONLY valid Python code and comments.

Example format:

```python
# Educational solution for the problem
import math

# Step 1: Parse the problem
# This problem involves [mathematical concept], which works by...
# [detailed explanation of the concept and approach]
...

# Step 2: Implement the solution
# We're using [technique] because...
# An alternative approach would be [alternative], but we chose this because...
...

# Calculate and print the final answer
result = ...
print(result)  # Just the number, no text
</response>"""

# List of all programmer prompts
PROGRAMMER_PROMPTS: List[str] = [
    PROGRAMMER_SYSTEM_PROMPT,
    EFFICIENT_ALGORITHM_PROMPT,
    NUMERICAL_STABILITY_PROMPT,
    TEST_DRIVEN_PROMPT,
    MATH_LIBRARY_PROMPT,
    EDUCATIONAL_CODE_PROMPT
]
