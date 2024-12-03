from typing import Dict
from langchain_core.messages import HumanMessage
class AnalysisAgent:
    """Agent that provides problem analysis and approach"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str) -> str:
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
        return await self.model.ainvoke(prompt)

class NextStepAgent:
    """Agent that provides the next step in a solution"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, current_solution: str = "") -> str:
        """Generate the next solution step"""
        input_text = (
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
            
        prompt = [
            HumanMessage(content=(
                "You are a mathematical solution expert focused on providing clear, detailed solution steps.\n\n" +
                input_text
            ))
        ]
        return await self.model.ainvoke(prompt)

class CompletionAgent:
    """Agent that completes partial solutions"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str, partial_solution: str) -> str:
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
        return await self.model.ainvoke(prompt)

class FullSolutionAgent:
    """Agent that provides complete solutions with analysis and steps"""
    
    def __init__(self, model):
        self.model = model
        
    async def generate(self, problem: str) -> str:
        """Generate a complete solution with analysis and steps"""
        prompt = [
            HumanMessage(content=(
                "You are a comprehensive mathematical solution expert who provides thorough analysis and detailed solutions.\n\n"
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
        return await self.model.ainvoke(prompt)
