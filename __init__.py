import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Make key utilities available at the package level
from utils.model_utils import (
    get_model, get_model_response, 
    OpenRouterChat, CustomChat,
    TimeoutException, time_limit,
    async_retry
)

from utils.solution_utils import (
    extract_numeric_answer, is_answer_correct,
    extract_answer_from_solution, NumericVerifier,
    validate_completion, validate_step,
    split_into_steps, get_partial_solutions,
    has_thinking_section, extract_thinking_section,
    extract_response_section, has_response_section,
    check_steps_status, has_boxed_answer,
    count_manual_steps, is_multiple_choice,
    STEP_NUMBER_PATTERNS
)

# Create a convenience module to make imports cleaner
from utils import model_utils, solution_utils
