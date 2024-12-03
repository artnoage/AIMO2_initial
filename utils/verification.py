import asyncio
from typing import Tuple, Optional
from utils.utils import extract_answer_from_solution, extract_numeric_answer
from utils.agents import AnswerVerifierAgent, SolutionVerifierAgent

async def verify_numeric(solution: str, correct_answer: str, tolerance: float = 1e-6) -> Tuple[Optional[float], bool]:
    """Verify solution using numeric comparison"""
    try:
        model_answer = extract_numeric_answer(solution)
        correct_float = float(correct_answer)
        
        if model_answer is None or not isinstance(model_answer, (int, float)):
            return None, False
            
        is_correct = abs(float(model_answer) - correct_float) <= tolerance
        return model_answer, is_correct
    except (ValueError, TypeError):
        return None, False

async def verify_solution_with_model(
    solution: str,
    correct_answer: str,
    problem: str,
    verifier_model,
    second_verifier_model=None
) -> Tuple[int, Optional[str]]:
    """
    Verify solution using model-based verification
    Returns:
    - verification_level (0-4)
    - extracted_answer or None
    
    Levels:
    0 - Failed format check
    1 - Failed answer verification
    2 - Failed first solution verification
    3 - Failed second verifier (if provided)
    4 - Passed all checks
    """
    model_answer = extract_answer_from_solution(solution)
    if model_answer is None or solution is None:
        return 0, None

    try:
        # Check answer equivalence
        answer_verifier = AnswerVerifierAgent(verifier_model)
        if not await answer_verifier.verify(problem, solution, correct_answer):
            return 1, model_answer

        # Check solution completeness
        solution_verifier = SolutionVerifierAgent(verifier_model)
        if not await solution_verifier.verify(problem, solution):
            return 2, model_answer
            
        # Only check second verifier if provided
        if second_verifier_model:
            second_solution_verifier = SolutionVerifierAgent(second_verifier_model)
            if not await second_solution_verifier.verify(problem, solution):
                return 3, model_answer
            
        return 4, model_answer

    except Exception as e:
        print(f"Verification error: {e}")
        return 0, None
import asyncio
from typing import Tuple, Optional
from utils.utils import extract_answer_from_solution, extract_numeric_answer
from utils.agents import AnswerVerifierAgent, SolutionVerifierAgent

async def verify_numeric(solution: str, correct_answer: str, tolerance: float = 1e-6) -> Tuple[Optional[float], bool]:
    """Verify solution using numeric comparison"""
    try:
        model_answer = extract_numeric_answer(solution)
        correct_float = float(correct_answer)
        
        if model_answer is None or not isinstance(model_answer, (int, float)):
            return None, False
            
        is_correct = abs(float(model_answer) - correct_float) <= tolerance
        return model_answer, is_correct
    except (ValueError, TypeError):
        return None, False

async def verify_solution_with_model(
    solution: str,
    correct_answer: str,
    problem: str,
    verifier_model,
    second_verifier_model=None
) -> Tuple[int, Optional[str]]:
    """
    Verify solution using model-based verification
    Returns:
    - verification_level (0-4)
    - extracted_answer or None
    
    Levels:
    0 - Failed format check
    1 - Failed answer verification
    2 - Failed first solution verification
    3 - Failed second verifier (if provided)
    4 - Passed all checks
    """
    model_answer = extract_answer_from_solution(solution)
    if model_answer is None or solution is None:
        return 0, None

    try:
        # Check answer equivalence
        answer_verifier = AnswerVerifierAgent(verifier_model)
        if not await answer_verifier.verify(problem, solution, correct_answer):
            return 1, model_answer

        # Check solution completeness
        solution_verifier = SolutionVerifierAgent(verifier_model)
        if not await solution_verifier.verify(problem, solution):
            return 2, model_answer
            
        # Only check second verifier if provided
        if second_verifier_model:
            second_solution_verifier = SolutionVerifierAgent(second_verifier_model)
            if not await second_solution_verifier.verify(problem, solution):
                return 3, model_answer
            
        return 4, model_answer

    except Exception as e:
        print(f"Verification error: {e}")
        return 0, None
