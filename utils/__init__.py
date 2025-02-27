import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Make key utilities available at the package level
from utils.model_utils import (
    get_model, get_model_response, 
    OpenRouterChat, CustomChat,
    TimeoutException, time_limit
)

from utils.solution_utils import (
    extract_numeric_answer, is_answer_correct,
    extract_answer_from_solution, NumericVerifier,
    validate_completion, validate_step,
    split_into_steps, get_partial_solutions
)
