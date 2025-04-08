from typing import List

# Original full solution prompt
FULLSOLUTION_SYSTEM_PROMPT = """You will be given a mathematical problem. Carefully analyze it before providing a well-structured response.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
Use this area as your creative scratchpad.
Freely document your thoughts, abstractions, corrections, and insights.
</thinking>

<response>
<step>Step 1: Clearly state initial calculations
Show work with LaTeX notation</step>

<step>Step 2: Logical next step
Clearly numbered and self-contained</step>

<step>Step N: Final conclusion clearly stated
Answer in \boxed{}</step>
</response>"""

# Variation 1: Detailed thinking section
DETAILED_THINKING_PROMPT = """You will be given a mathematical problem.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
1. Define key variables clearly.
2. List relevant formulas or theorems.
3. Outline step-by-step solution plan.
4. Identify and address potential challenges.
5. Verify logical consistency.
</thinking>

<response>
<step>Step 1: Initial calculations
LaTeX notation</step>

<step>Step 2: Next step logically derived</step>

<step>Step N: Final answer in \boxed{}</step>
</response>"""

# Variation 2: Mathematical rigor
RIGOROUS_PROMPT = """Solve the mathematical problem with rigorous mathematical logic.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
1. Precise definitions and notation.
2. Justify each theorem or formula.
3. Logical step-by-step reasoning.
</thinking>

<response>
<step>Step 1: Define variables precisely
Use rigorous notation</step>

<step>Step 2: Clearly justified logical steps</step>

<step>Step N: Conclude rigorously in \boxed{}</step>
</response>"""

# Variation 3: Multiple solution approaches
MULTIPLE_APPROACHES_PROMPT = """Solve the mathematical problem by exploring multiple methods.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
1. Present multiple viable approaches.
2. Discuss advantages of each.
3. Select best approach for final solution.
</thinking>

<response>
<step>Step 1: Initial chosen approach
LaTeX notation</step>

<step>Step 2: Logical next step</step>

<step>Step N: Final answer clearly in \boxed{}</step>
</response>"""

# Variation 4: Educational and explanatory
EDUCATIONAL_PROMPT = """Solve the mathematical problem with detailed educational explanations.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
Clearly document reasoning, educational insights, and intuitive explanations.
</thinking>

<response>
<step>Step 1: Introduce key concepts</step>

<step>Step 2: Explain solution clearly</step>

<step>Step N: Summarize clearly, final answer \boxed{}</step>
</response>"""

# Variation 5: Concise and direct
CONCISE_PROMPT = """Solve the mathematical problem in a concise, direct manner.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
Briefly note key insights and simplest method.
</thinking>

<response>
<step>Step 1: Core calculations concisely stated</step>

<step>Step N: Efficiently reach final answer \boxed{}</step>
</response>"""

# Variation 6: Creative exploration
CREATIVE_PROMPT = """Solve the mathematical problem through creative exploration.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
Explore unconventional methods and document creative insights.
</thinking>

<response>
<step>Step 1: Creative initial insight</step>

<step>Step N: Innovative final conclusion \boxed{}</step>
</response>"""

# Variation 7: Emphasis on clarity
CLARITY_PROMPT = """Solve the mathematical problem with maximum clarity.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
Prioritize clarity in explaining logic and choice of method.
</thinking>

<response>
<step>Step 1: Clearly articulated initial steps</step>

<step>Step N: Clearly presented final answer \boxed{}</step>
</response>"""

# Variation 8: Error anticipation and handling
ERROR_ANTICIPATION_PROMPT = """Solve the mathematical problem anticipating and addressing possible errors.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
Identify potential errors and clearly outline your plan to address them.
</thinking>

<response>
<step>Step 1: Carefully stated initial calculations
Include error checks</step>

<step>Step N: Verified final solution \boxed{}</step>
</response>"""

# Variation 9: Real-world contextualization
REAL_WORLD_PROMPT = """Solve the mathematical problem highlighting real-world context or applications.
Your output must include two clearly separated sections: a **thinking** section and a **response** section.

<thinking>
Relate problem clearly to real-world scenarios and applications.
</thinking>

<response>
<step>Step 1: Contextual introduction</step>

<step>Step N: Contextually relevant final answer \boxed{}</step>
</response>"""

# List of updated solution prompts
SOLUTION_PROMPTS: List[str] = [
    FULLSOLUTION_SYSTEM_PROMPT,
    DETAILED_THINKING_PROMPT,
    RIGOROUS_PROMPT,
    MULTIPLE_APPROACHES_PROMPT,
    EDUCATIONAL_PROMPT,
    CONCISE_PROMPT,
    CREATIVE_PROMPT,
    CLARITY_PROMPT,
    ERROR_ANTICIPATION_PROMPT,
    REAL_WORLD_PROMPT
]
