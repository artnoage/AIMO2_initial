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
            return 0, 4, None
            
        score = 0
        verification_steps = [
            ("Mathematical correctness", self.first_model, 
             "You are a strict mathematical validator. Your ONLY allowed responses are 'yes' or 'no'. Respond 'yes' if ALL mathematical steps and calculations are correct, otherwise 'no'."),
            ("Solution completeness", self.first_model,
             "You are a strict solution validator. Your ONLY allowed responses are 'yes' or 'no'. Respond 'yes' if the solution includes ALL necessary steps and explanations, otherwise 'no'."),
            ("Final answer verification", self.second_model,
             "You are a strict answer validator. Your ONLY allowed responses are 'yes' or 'no'. Respond 'yes' if the final answer is correctly derived and matches the solution steps, otherwise 'no'."),
            ("Logical coherence", self.second_model,
             "You are a strict logic validator. Your ONLY allowed responses are 'yes' or 'no'. Respond 'yes' if the solution flows logically and all steps connect properly, otherwise 'no'.")
        ]
        
        for step_name, model, system_content in verification_steps:
            prompt = [
                HumanMessage(content=f"Problem:\n{problem}\n\nProposed solution:\n{solution}\n\nVerification task - {step_name}: Is this aspect of the solution correct? Answer ONLY with 'yes' or 'no'.")
            ]
            response = await get_model_response(model, prompt)
            response_text = response.strip().lower()
            if response_text == 'yes':
                score += 1
                
        return score, 4, model_answer

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
