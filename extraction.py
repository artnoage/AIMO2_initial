from typing import List, Optional, Dict, Union, Any
from collections import defaultdict

def get_programming_majority(programming_answers: List[float], tolerance: float = 1e-2) -> Optional[float]:
    """
    Find the majority answer from a list of float values using tolerance-based grouping.
    
    Args:
        programming_answers: List of float answers
        tolerance: Numeric tolerance for grouping similar answers
        
    Returns:
        The majority answer as a float, or None if no answers
    """
    # Filter out None values
    valid_answers = [ans for ans in programming_answers if ans is not None]
    
    if not valid_answers:
        return None
    
    # Group answers by tolerance
    groups = defaultdict(list)
    
    for ans in valid_answers:
        # Find if this answer belongs to an existing group
        found_group = False
        for group_key in groups:
            if abs(ans - group_key) <= tolerance:
                groups[group_key].append(ans)
                found_group = True
                break
        
        # If not found in any group, create a new group
        if not found_group:
            groups[ans].append(ans)
    
    # Find the group with the most answers
    if groups:
        majority_group = max(groups.items(), key=lambda x: len(x[1]))
        return majority_group[0]  # Return the group key (representative value)
    
    return None

def get_hybrid_majority(programming_answers: List[float], standard_answers: List[float], tolerance: float = 1e-2) -> Optional[float]:
    """
    Find the majority answer using the hybrid approach that considers the intersection
    of programming and standard solution answers.
    
    Args:
        programming_answers: List of float answers from programming solutions
        standard_answers: List of float answers from standard solutions
        tolerance: Numeric tolerance for grouping similar answers
        
    Returns:
        The final answer as a float, or None if no valid answers
    """
    # Filter out None values
    valid_programming_answers = [ans for ans in programming_answers if ans is not None]
    valid_standard_answers = [ans for ans in standard_answers if ans is not None]
    
    # If either list is empty, return the majority from the non-empty list
    if not valid_programming_answers and not valid_standard_answers:
        return None
    elif not valid_programming_answers:
        return get_programming_majority(valid_standard_answers, tolerance)
    elif not valid_standard_answers:
        return get_programming_majority(valid_programming_answers, tolerance)
    
    # Find intersection of answers using numeric tolerance for comparison
    intersection_values = []
    
    # For each standard answer, find programming answers that are within tolerance
    for std_ans in valid_standard_answers:
        for prog_ans in valid_programming_answers:
            if abs(std_ans - prog_ans) <= tolerance:
                # Add both values to the intersection
                intersection_values.append(std_ans)
                intersection_values.append(prog_ans)
    
    # If we have intersection values, find the majority among them
    if intersection_values:
        return get_programming_majority(intersection_values, tolerance)
    
    # If no intersection, use the most common from either method
    programming_majority = get_programming_majority(valid_programming_answers, tolerance)
    standard_majority = get_programming_majority(valid_standard_answers, tolerance)
    
    # Return programming majority if available, otherwise standard majority
    return programming_majority if programming_majority is not None else standard_majority


def get_closest_integer(value: float) -> int:
    """
    Find the closest integer to a given float value.
    
    Args:
        value: Float value to round to nearest integer
        
    Returns:
        The closest integer to the input value
    """
    return round(value)
