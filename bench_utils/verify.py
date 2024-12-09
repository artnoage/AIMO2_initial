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
            

        # Try to convert both answers to float with debug info
        numeric_answer, model_error = extract_numeric_answer(solution, debug=True)
        if numeric_answer is None:
            return 0, 1, f"{model_answer} (Error: {model_error})"
            
        # First try to convert the correct answer
        correct_answer_num, correct_error = extract_numeric_answer(correct_answer, debug=True)
        if correct_answer_num is not None:
            try:
                correct_float = float(correct_answer.strip())
            except (ValueError, TypeError, AttributeError) as e:
                # If correct answer isn't numeric, return the raw answer string and mark as wrong
                return 0, 1, f"{model_answer} (Error: Correct answer parse failed - {str(e)})"
        else:
            correct_float = correct_answer_num

        # Then try to convert the numeric answer
        if not isinstance(numeric_answer, (int, float)):
            # If model answer isn't numeric, return it as string and mark as wrong
            return 0, 1, f"{model_answer} (Error: Not a number - {model_error})"

        # Only compare if both are valid numbers
        is_correct = abs(float(numeric_answer) - correct_float) <= self.tolerance
        if not is_correct and model_error:
            model_answer = f"{model_answer} (Debug: {model_error})"
        return 1 if is_correct else 0, 1, model_answer + "and the numeric" +numeric_answer
                
 

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
