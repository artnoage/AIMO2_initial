import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Make key utilities available at the package level
from utils.model_utils import get_model, get_model_response
from utils.solution_utils import extract_answer_from_solution, extract_numeric_answer
