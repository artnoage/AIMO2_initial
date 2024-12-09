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

        
        # Extract and convert model answer
        numeric_answer, model_error = extract_numeric_answer(model_answer, debug=True)
        if numeric_answer is None:
            return 0, 1, f"{model_answer} (Error: {model_error})"
        # Extract and convert correct answer
        correct_numeric, correct_error = extract_numeric_answer(correct_answer, debug=True)
        if correct_numeric is None:
            return 0, 1, f"{model_answer} (Error: Correct answer not parseable - {correct_error})"

        # Compare the numeric values
        is_correct = abs(numeric_answer - correct_numeric) <= self.tolerance
        
        # Include both parsing details in debug info
        display_answer = f"{model_answer} (Model: {model_error}, Correct: {correct_error})"
            
        return 1 if is_correct else 0, 1, display_answer
                
 

class AnswerVerifier(BaseVerifier):
    def __init__(self, model):
        self.model = model
        
    async def verify(self, solution: str, correct_answer: str, problem: str) -> Tuple[int, int, Optional[str]]:
        try:
            model_answer = extract_answer_from_solution(solution)
            if model_answer is None or solution is None:
                return 0, 1, None
                
            prompt = [
            SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
            HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
        ]
            
            try:
                response = await get_model_response(self.model, prompt)
                response = response.strip().lower()
                if response not in ['yes', 'no']:
                    print(f"Warning: Unexpected verification response: {response}")
                    return 0, 1, model_answer
                is_correct = response == 'yes'
                return 1 if is_correct else 0, 1, model_answer
            except Exception as e:
                print(f"Warning: Verification API error: {str(e)}")
                return 0, 1, model_answer
                
        except Exception as e:
            print(f"Warning: General verification error: {str(e)}")
            return 0, 1, None

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
            SystemMessage(content="You are a mathematical answer validator. Given a problem and two answers, respond ONLY with 'yes' if they are mathematically equivalent, or 'no' if they are different. Just one word, no explanation."),
            HumanMessage(content=f"Problem:\n{problem}\n\nAre these two answers equivalent?\nAnswer 1: {model_answer}\nAnswer 2: {correct_answer}")
        ]
        
        response = await get_model_response(self.first_model, answer_prompt)
        if response.strip().lower() != 'yes':
            return score, total_steps, model_answer
        score += 1
        
        # Step 2: First model solution validation
        solution_prompt = [
            SystemMessage(content="You are a mathematical solution validator. Given a problem and a proposed solution, respond ONLY with 'yes' if the solution is mathematically correct, detailed and coherent, or 'no' if it contains any errors, lacks detail, or has incoherent reasoning. Just one word, no explanation."),
            HumanMessage(content=f"Problem:\n{problem}\n\nProposed solution:\n{solution}\n\nIs this solution mathematically correct and complete?")
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
