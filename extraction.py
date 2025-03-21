from typing import List, Optional, Tuple, Dict, Union, Any

def calculate_answer_majority(answers, tolerance=1e-2):
    """
    Calculate the most common answer by counting how many answers are within tolerance
    of each unique answer.
    
    Args:
        answers: List of answers (can be numeric or string)
        tolerance: Numeric tolerance for grouping similar answers
        
    Returns:
        Tuple of (majority_answer, count_dict) where count_dict maps each answer to its count
    """
    if not answers or all(ans is None for ans in answers):
        return None, {}
    
    # Filter out None values
    valid_answers = [ans for ans in answers if ans is not None]
    
    # Convert to numeric where possible
    numeric_answers = []
    for ans in valid_answers:
        try:
            if isinstance(ans, (int, float)):
                numeric_answers.append((ans, str(ans)))
            else:
                numeric_val, _ = extract_numeric_answer(str(ans))
                if numeric_val is not None:
                    numeric_answers.append((numeric_val, str(ans)))
                else:
                    # Keep non-numeric answers as is
                    numeric_answers.append((None, str(ans)))
        except:
            numeric_answers.append((None, str(ans)))
    
    # Count how many answers are within tolerance of each answer
    count_dict = {}
    for i, (num_val, str_val) in enumerate(numeric_answers):
        # Initialize count for this answer
        if str_val not in count_dict:
            count_dict[str_val] = 0
        
        # Count all answers within tolerance of this one
        for other_num, other_str in numeric_answers:
            if num_val is not None and other_num is not None:
                # Both are numeric, use tolerance
                if abs(num_val - other_num) <= tolerance:
                    count_dict[str_val] += 1
            else:
                # At least one is non-numeric, use exact string matching
                if str_val == other_str:
                    count_dict[str_val] += 1
    
    # Find the answer with the highest count
    if count_dict:
        majority_answer = max(count_dict.items(), key=lambda x: x[1])[0]
        return majority_answer, count_dict
    else:
        return None, {}

def extract_numeric_answer(answer: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value from a LaTeX answer string.
    First tries to evaluate using sympy, then falls back to direct float conversion.
    Returns float if found, None otherwise.
    """
    import re
    
    # Simple implementation for the example
    try:
        # Try direct conversion
        return float(answer), None
    except (ValueError, TypeError):
        # Try to extract numbers from the string
        number_match = re.search(r'[-+]?\d*\.?\d+', answer)
        if number_match:
            try:
                return float(number_match.group(0)), None
            except (ValueError, TypeError):
                pass
    return None, "Could not extract numeric value"

def get_programming_majority(programming_answers: List[Union[float, int, str, None]], tolerance: float = 1e-2) -> Optional[float]:
    """
    Find the majority answer from programming solutions using the tolerance-based approach.
    
    Args:
        programming_answers: List of answers from programming solutions
        tolerance: Numeric tolerance for grouping similar answers
        
    Returns:
        The majority answer as a float if possible, otherwise as a string or None
    """
    # Use the calculate_answer_majority function to find the majority
    majority_answer, _ = calculate_answer_majority(programming_answers, tolerance)
    
    # Try to convert to float if possible
    if majority_answer is not None:
        try:
            if isinstance(majority_answer, (int, float)):
                return float(majority_answer)
            else:
                numeric_val, _ = extract_numeric_answer(str(majority_answer))
                if numeric_val is not None:
                    return numeric_val
        except:
            pass
    
    # If we couldn't convert to float, return the original majority answer
    return majority_answer

def get_hybrid_majority(programming_answers: List[Union[float, int, str, None]], 
                        standard_answers: List[Union[float, int, str, None]], 
                        tolerance: float = 1e-2) -> Optional[float]:
    """
    Find the majority answer using the hybrid approach that considers the intersection
    of programming and standard solution answers.
    
    Args:
        programming_answers: List of answers from programming solutions
        standard_answers: List of answers from standard solutions
        tolerance: Numeric tolerance for grouping similar answers
        
    Returns:
        The final answer as a float if possible, otherwise as a string or None
    """
    # Find intersection of answers using numeric tolerance for comparison
    intersection_answers = set()
    
    # For each standard answer, find programming answers that are within tolerance
    for std_ans in standard_answers:
        if std_ans is None:
            continue
            
        # Try to convert to numeric for comparison
        std_numeric = None
        try:
            if isinstance(std_ans, (int, float)):
                std_numeric = std_ans
            else:
                std_numeric, _ = extract_numeric_answer(str(std_ans))
        except:
            pass
            
        # If numeric, compare with tolerance
        if std_numeric is not None:
            for prog_ans in programming_answers:
                if prog_ans is None:
                    continue
                    
                # Try to convert to numeric
                prog_numeric = None
                try:
                    if isinstance(prog_ans, (int, float)):
                        prog_numeric = prog_ans
                    else:
                        prog_numeric, _ = extract_numeric_answer(str(prog_ans))
                except:
                    pass
                    
                # Compare with tolerance if both are numeric
                if prog_numeric is not None:
                    if abs(std_numeric - prog_numeric) <= tolerance:
                        # Add both original string representations to the intersection
                        intersection_answers.add(str(std_ans))
                        intersection_answers.add(str(prog_ans))
        else:
            # For non-numeric answers, use exact string comparison
            if str(std_ans) in {str(ans) for ans in programming_answers if ans is not None}:
                intersection_answers.add(str(std_ans))
    
    # Determine final answer based on intersection
    final_answer = None
    if intersection_answers:
        # If there's an intersection, use the calculate_answer_majority approach to calculate the majority answer
        # from all answers, but filter to only include those in the intersection
        all_answers = programming_answers + standard_answers
        _, all_answer_counts = calculate_answer_majority(all_answers, tolerance=tolerance)
        
        # Filter to only include answers in the intersection
        intersection_counts = {}
        for ans_str, count in all_answer_counts.items():
            # Check if this answer is in the intersection
            in_intersection = False
            
            # Direct string match
            if ans_str in intersection_answers:
                in_intersection = True
            else:
                # Check numeric tolerance match
                ans_numeric = None
                try:
                    ans_numeric, _ = extract_numeric_answer(ans_str)
                except:
                    pass
                    
                if ans_numeric is not None:
                    for ia in intersection_answers:
                        ia_numeric = None
                        try:
                            ia_numeric, _ = extract_numeric_answer(ia)
                        except:
                            pass
                            
                        if ia_numeric is not None and abs(ans_numeric - ia_numeric) <= tolerance:
                            in_intersection = True
                            break
            
            if in_intersection:
                intersection_counts[ans_str] = count
        
        if intersection_counts:
            final_answer = max(intersection_counts.items(), key=lambda x: x[1])[0]
    
    if final_answer is None:
        # If no intersection or no final answer determined, use the most common from either method
        programming_majority_answer, _ = calculate_answer_majority(programming_answers, tolerance)
        standard_majority_answer, _ = calculate_answer_majority(standard_answers, tolerance)
        final_answer = programming_majority_answer if programming_majority_answer else standard_majority_answer
    
    # Try to convert to float if possible
    if final_answer is not None:
        try:
            if isinstance(final_answer, (int, float)):
                return float(final_answer)
            else:
                numeric_val, _ = extract_numeric_answer(str(final_answer))
                if numeric_val is not None:
                    return numeric_val
        except:
            pass
    
    # If we couldn't convert to float, return the original final answer
    return final_answer
