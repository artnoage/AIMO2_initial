from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from bench_utils.benchmark_utils import get_model_response

BENCHMARK_SYSTEM_PROMPT = """You are a mathematical problem solver. Your task is to solve mathematical problems step by step.

Guidelines:
1. Read the problem carefully
2. Show your work clearly with numbered steps
3. Use LaTeX notation for mathematical expressions
4. Explain your reasoning in [brackets]
5. End with a final answer in \boxed{}
"""

NUMERIC_SOLVER_SYSTEM_PROMPT = """You are a mathematical problem solver focused on numerical answers. Your task is to solve mathematical 
problems and provide precise numeric solutions.

Guidelines:
1. Read the problem carefully
2. Show your work clearly with numbered steps
3. Use LaTeX notation for mathematical expressions
4. Explain your reasoning in [brackets]
5. End with a final numeric answer in \boxed{}
6. Ensure your answer is a precise number
"""

ANSWER_VERIFIER_SYSTEM_PROMPT = """You are a mathematical solution verifier. Your task is to verify if two solutions are equivalent.

Guidelines:
1. Compare the final answers carefully
2. Consider mathematical equivalence, not just exact matches
3. Account for different forms of the same answer
4. Be precise in your verification
5. Return True only if answers are mathematically equivalent
"""
class AnalysisAgent:
    """Agent that provides problem analysis and approach"""
    
    def __init__(self, model, numeric: bool = False):
        self.model = model
        self.system_prompt = NUMERIC_SOLVER_SYSTEM_PROMPT if numeric else BENCHMARK_SYSTEM_PROMPT
        
    async def generate(self, problem: str, running_id: int = 0, attempt: int = 0) -> str:
        """Generate analysis for a given problem"""
        prompt = [
            HumanMessage(content=(
                "You are a mathematical analysis expert. Your role is to analyze problems "
                "and outline solution approaches without solving them.\n\n"
                f"Here is a mathematical problem:\n\n{problem}\n\n"
                "Before solving this problem step-by-step, provide a thorough analysis that:\n"
                "1. Categorizes the problem type\n"
                "2. Lists the specific theorems and techniques that will be useful\n"
                "3. Outlines the general approach to solving it\n\n"
                "Important guidelines:\n"
                "- Start with '**Problem Analysis and Approach**:'\n"
                "- Be specific about which theorems/techniques apply\n"
                "- Explain why these approaches are suitable\n"
                "- Do NOT provide the actual solution steps\n\n"
                "Please provide the analysis:"
            ))
        ]
        return await get_model_response(self.model, prompt)

class NextStepAgent:
    """Agent that provides the next step in a solution"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, current_solution: str = "", running_id: int = 0, attempt: int = 0, timeout: int = 300) -> str:
        """Generate the next solution step"""
        input_text = (
            "You are a mathematical solution expert focused on providing clear, detailed solution steps.\n\n"
            f"Here is a mathematical problem:\n\n{problem}\n\n"
            "Your task is to provide the next step in the solution. "
            "Make sure your step is detailed and mathematically rigorous.\n\n"
            "Guidelines:\n"
            "- Provide exactly ONE step\n"
            "- Include clear explanations\n"
            "- Use LaTeX notation where appropriate\n"
            "- Include justification in [brackets]\n"
            "- Number your step appropriately\n"
        )
            
        if current_solution:
            input_text += f"\nHere are the steps so far:\n\n{current_solution}\n\nProvide the next step:"
        else:
            input_text += "\nStart the solution with Step 1:"
            
        prompt = [HumanMessage(content=input_text)]
        return await get_model_response(self.model, prompt)

class CompletionAgent:
    """Agent that completes partial solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, partial_solution: str, running_id: int = 0, attempt: int = 0) -> str:
        """Complete a partial solution"""
        prompt = [
            HumanMessage(content=(
                "You are a mathematical solution expert who excels at completing partial solutions.\n\n"
                f"Here is a mathematical problem:\n\n{problem}\n\n"
                "I will show you the beginning of a step-by-step mathematical solution. "
                "Your task is to complete the solution by continuing with the same style and rigor.\n\n"
                "Important guidelines:\n"
                "- Maintain the same level of detail and explanation as the previous steps\n"
                "- Continue the step numbering sequence\n"
                "- Use LaTeX notation consistently\n"
                "- Provide justification for each step in [brackets]\n"
                "- End with a clear boxed answer using \\boxed{}\n\n"
                f"Here is the partial solution:\n\n{partial_solution}\n\n"
                "Please complete the remaining steps following the same format:"
            ))
        ]
        return await get_model_response(self.model, prompt)


class FullSolutionAgent:
    """Agent that provides complete solutions with analysis and steps"""
    
    def __init__(self, model, numeric: bool = False):
        self.model = model
        self.system_prompt = NUMERIC_SOLVER_SYSTEM_PROMPT if numeric else BENCHMARK_SYSTEM_PROMPT
        
    async def generate(self, problem: str, running_id: int = 0, attempt: int = 0) -> str:
        """Generate a complete solution with analysis and steps"""
        prompt = [
            HumanMessage(content=(
                f"{self.system_prompt}\n\n"
                f"Here is a mathematical problem to solve:\n\n{problem}\n\n"
                "Please provide a complete solution following these guidelines:\n"
                "1. Start with '**Problem Analysis and Approach**:' section explaining:\n"
                "   - Problem type and key concepts involved\n"
                "   - Relevant theorems and techniques\n"
                "   - Overall solution strategy\n\n"
                "2. Then provide a detailed step-by-step solution:\n"
                "   - Number each step clearly (Step 1, Step 2, etc.)\n"
                "   - Show all work and intermediate calculations\n"
                "   - Use LaTeX notation for mathematical expressions\n"
                "   - Provide justification in [brackets] for key steps\n"
                "   - End with final answer in \\boxed{}\n\n"
                "Please solve the problem completely:"
            ))
        ]
        return await get_model_response(self.model, prompt)
