from typing import Optional, Tuple, Literal
from abc import ABC, abstractmethod
from langchain_core.messages import SystemMessage, HumanMessage
from bench_utils.benchmark_utils import *


VerificationType = Literal['numeric', 'answer', 'solution']

class BaseVerifier(ABC):
    @abstractmethod
    async def verify(self, solution: str, correct_answer: str, problem: str) -> Tuple[int, int, Optional[str]]:
        """
        Verify solution and return score, total steps, and extracted answer
        
        Returns:
        - score: Number of successful verification steps
        - total_steps: Total number of verification steps
        - extracted_answer or None
        """
        pass

class NumericVerifier(BaseVerifier):
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    async def verify(self, solution: str, correct_answer: str, problem: str) -> Tuple[int, int, Optional[str]]:
        if not solution or not correct_answer:
            return 0, 1, None
            
        model_answer = extract_answer_from_solution(solution)
        if model_answer is None:
            return 0, 1, None
            
        try:
            # Try to convert both answers to float
            numeric_answer = extract_numeric_answer(solution)
            if numeric_answer is None:
                return 0, 1, model_answer
                
            # First try to convert the correct answer
            try:
                correct_float = float(correct_answer.strip())
            except (ValueError, TypeError, AttributeError):
                # If correct answer isn't numeric, return the raw answer string and mark as wrong
                return 0, 1, model_answer

            # Then try to convert the numeric answer
            if not isinstance(numeric_answer, (int, float)):
                # If model answer isn't numeric, return it as string and mark as wrong
                return 0, 1, model_answer

            # Only compare if both are valid numbers
            is_correct = abs(float(numeric_answer) - correct_float) <= self.tolerance
            return 1 if is_correct else 0, 1, model_answer
                
        except Exception as e:
            print(f"Warning: Verification error: {e}")
            return 0, 1, model_answer

class AnswerVerifier(BaseVerifier):
    def __init__(self, model):
        self.model = model
        
    async def verify(self, solution: str, correct_answer: str, problem: str) -> Tuple[int, int, Optional[str]]:
        model_answer = extract_answer_from_solution(solution)
        if model_answer is None or solution is None:
            return 0, 1, None
            
        prompt = [
            SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
            HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
        ]
        
        response = await get_model_response(self.model, prompt)
        is_correct = response.strip().lower() == 'yes'
        return 1 if is_correct else 0, 1, model_answer

class SolutionVerifier(BaseVerifier):
    def __init__(self, first_model, second_model):
        self.first_model = first_model
        self.second_model = second_model
        
    async def verify(self, solution: str, correct_answer: str, problem: str) -> Tuple[int, int, Optional[str]]:
        model_answer = extract_answer_from_solution(solution)
        if model_answer is None or solution is None:
            return 0, 3, None
            
        score = 0
        total_steps = 3  # Answer check + two solution validations
        
        # Step 1: Check if the answer is correct
        answer_prompt = [
            SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
            HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
        ]
        
        response = await get_model_response(self.first_model, answer_prompt)
        if response.strip().lower() != 'yes':
            return score, total_steps, model_answer
        score += 1
        
        # Step 2: First model solution validation
        solution_prompt = [
            SystemMessage(content="You are a mathematical solution validator. Verify if the solution is complete, correct, and well-explained. Respond ONLY with 'yes' or 'no'."),
            HumanMessage(content=f"Problem:\n{problem}\n\nSolution:\n{solution}\n\nIs this solution mathematically correct and complete? Answer ONLY with 'yes' or 'no'.")
        ]
        
        response = await get_model_response(self.first_model, solution_prompt)
        if response.strip().lower() != 'yes':
            return score, total_steps, model_answer
        score += 1
        
        # Step 3: Second model solution validation
        response = await get_model_response(self.second_model, solution_prompt)
        if response.strip().lower() == 'yes':
            score += 1
            
        return score, total_steps, model_answer

def create_verifier(verification_type: str, **kwargs) -> BaseVerifier:
    """Factory function to create appropriate verifier"""
    if verification_type == 'numeric':
        return NumericVerifier(tolerance=kwargs.get('tolerance', 1e-6))
    elif verification_type == 'answer':
        if 'verifier_model' not in kwargs:
            raise ValueError("Verifier model required for answer verification")
        return AnswerVerifier(kwargs['verifier_model'])
    elif verification_type == 'solution':
        if 'verifier_model' not in kwargs or 'second_verifier_model' not in kwargs:
            raise ValueError("Both verifier models required for solution verification")
        return SolutionVerifier(kwargs['verifier_model'], kwargs['second_verifier_model'])
    else:
        raise ValueError(f"Unknown verification type: {verification_type}")
