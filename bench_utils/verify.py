from typing import Optional, Tuple
from bench_utils.benchmark_utils import *

class NumericVerifier:
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        
    async def verify(self, solution: str, correct_answer: str, problem: str) -> Tuple[int, int, Optional[str]]:
        if not solution or not correct_answer:
            return 0, 1, None
            
        model_answer = extract_answer_from_solution(solution)
        if model_answer is None:
            return 0, 1, None
        
        # Extract and convert model answer
        numeric_answer, model_error = extract_numeric_answer(model_answer, debug=False)
        if numeric_answer is None:
            return 0, 1, model_answer if model_error is None else f"{model_answer} (Error: {model_error})"
            
        # Extract and convert correct answer
        correct_numeric, correct_error = extract_numeric_answer(correct_answer, debug=False)
        if correct_numeric is None:
            return 0, 1, model_answer if correct_error is None else f"{model_answer} (Error: Correct answer not parseable - {correct_error})"

        # Compare the numeric values
        is_correct = abs(numeric_answer - correct_numeric) <= self.tolerance
        
        # Return just the model answer if no debug info, otherwise include parsing details
        display_answer = model_answer if model_error is None else f"{model_answer} (Model: {model_error}, Correct: {correct_error})"
            
        return 1 if is_correct else 0, 1, display_answer
