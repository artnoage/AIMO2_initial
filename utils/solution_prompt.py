"""
Collection of solution prompts for mathematical problem-solving.
These are variations of the full solution system prompt.
"""

from typing import List

# Original full solution prompt
FULLSOLUTION_SYSTEM_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.
Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
Use this area as your creative scratchpad.
Feel free to capture your thoughts, abstractions, corrections, or ideas in any order and form you wish—without constraints. 
Use this freedom to ensure you've gathered all insights necessary to clearly and effectively provide the requested response.
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

# Variation 1: More detailed thinking section guidance
DETAILED_THINKING_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.
Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
In this section, organize your approach to the problem:
1. Identify the key variables and what you're solving for
2. Consider relevant formulas, theorems, or properties
3. Plan your solution strategy step by step
4. Anticipate potential challenges and how to address them
5. Double-check your approach for logical consistency
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

# Variation 2: Emphasizing mathematical rigor
RIGOROUS_PROMPT = """You will be given a mathematical problem. Analyze it with mathematical rigor before providing a well-structured response.
Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
Use this area as your creative scratchpad.
Feel free to capture your thoughts, abstractions, corrections, or ideas in any order and form you wish—without constraints. 
Use this freedom to ensure you've gathered all insights necessary to clearly and effectively provide the requested response.
</thinking>

<response>
<step>Step 1: Begin by defining all variables and clarifying the problem statement
Use precise mathematical notation and definitions</step>

<step>Step 2: Apply relevant theorems, properties, or formulas
Justify each application with a brief explanation</step>

<step>Step 3: Continue with subsequent logical steps
Ensure each step follows rigorously from previous ones</step>

<step>Step N: In your final step, state your conclusion
Put your final answer in \\boxed{} and verify it satisfies all constraints</step>
</response>
"""

# Variation 3: Encouraging multiple solution approaches
MULTIPLE_APPROACHES_PROMPT = """You will be given a mathematical problem. Consider multiple approaches before providing a well-structured response.
Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
In this section:
1. Consider at least two different approaches to solve the problem
2. Evaluate the advantages and limitations of each approach
3. Select the most elegant or efficient approach for your final solution
4. If relevant, note any alternative methods that could work
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

# Variation 4: Educational focus with explanations
EDUCATIONAL_PROMPT = """You will be given a mathematical problem. Solve it with clear explanations as if teaching a student.
Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
Use this area as your creative scratchpad.
Feel free to capture your thoughts, abstractions, corrections, or ideas in any order and form you wish—without constraints. 
Use this freedom to ensure you've gathered all insights necessary to clearly and effectively provide the requested response.
</thinking>

<response>
<step>Step 1: Begin by explaining the key concepts needed for this problem
Introduce relevant formulas or theorems with brief explanations</step>

<step>Step 2: Start solving the problem with clear reasoning
Explain why you're taking each approach and what it accomplishes</step>

<step>Step 3: Continue with subsequent steps
Highlight important techniques or insights along the way</step>

<step>Step N: In your final step, state your conclusion
Put your final answer in \\boxed{} and summarize the key takeaways</step>
</response>
"""

# Variation 5: Concise and direct approach
CONCISE_PROMPT = """You will be given a mathematical problem. Solve it efficiently with minimal steps.
Your output must include two clearly separated sections: **Thinking** and **Response**.

<thinking>
Use this area as your creative scratchpad.
Feel free to capture your thoughts, abstractions, corrections, or ideas in any order and form you wish—without constraints. 
Use this freedom to ensure you've gathered all insights necessary to clearly and effectively provide the requested response.
</thinking>

<response>
<step>Step 1: Identify the core problem and select the most direct approach
Use appropriate mathematical notation</step>

<step>Step 2: Apply the chosen method efficiently
Skip trivial steps while maintaining clarity</step>

<step>Step N: In your final step, state your conclusion
Put your final answer in \\boxed{}</step>
</response>
"""

# List of all solution prompts
SOLUTION_PROMPTS: List[str] = [
    FULLSOLUTION_SYSTEM_PROMPT,
    DETAILED_THINKING_PROMPT,
    RIGOROUS_PROMPT,
    MULTIPLE_APPROACHES_PROMPT,
    EDUCATIONAL_PROMPT,
    CONCISE_PROMPT
]
