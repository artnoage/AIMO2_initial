import os
import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
import sys
import torch
from collections import Counter
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.model_utils import *
from utils.solution_utils import *
from utils.agents import *
from utils.logger import BenchmarkLogger
from utils.similarity_checker import SolutionSimilarityChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

def calculate_confidence_boosted_similarity(solutions: List[Dict], similarity_matrix: torch.Tensor) -> Tuple[Optional[str], float]:
    """
    Calculate the majority answer using a confidence-boosted similarity approach.
    
    This method combines:
    1. Solution similarity (how similar a solution's thinking is to others)
    2. Solution confidence signals (length of reasoning, verification steps, etc.)
    3. Answer consistency (how many solutions arrived at the same answer)
    
    Args:
        solutions: List of solution dictionaries with 'answer' and 'solution' keys
        similarity_matrix: Tensor of pairwise similarities between solutions
        
    Returns:
        Tuple of (majority_answer, confidence_score)
    """
    # Filter out None answers
    valid_indices = [i for i, s in enumerate(solutions) if s['answer'] is not None]
    if not valid_indices:
        return None, 0.0
    
    # Extract valid solutions
    valid_solutions = [solutions[i] for i in valid_indices]
    
    # Calculate similarity component: average similarity to other solutions
    similarity_scores = []
    for i in valid_indices:
        similarities = [similarity_matrix[i, j].item() for j in valid_indices if i != j]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        similarity_scores.append(avg_similarity)
    
    # Calculate confidence component based on solution characteristics
    confidence_scores = []
    for i, idx in enumerate(valid_indices):
        solution = solutions[idx]
        thinking = solution['thinking']
        
        # Factors that might indicate confidence:
        # 1. Length of reasoning (longer reasoning often indicates more thorough work)
        length_factor = min(1.0, len(thinking) / 1000)  # Cap at 1.0
        
        # 2. Presence of verification steps (checking work indicates confidence)
        verification_factor = 0.2 if "verify" in thinking.lower() or "check" in thinking.lower() else 0.0
        
        # 3. Structured approach (step-by-step solutions tend to be more reliable)
        structure_factor = 0.2 if thinking.count("\n") > 5 else 0.0
        
        # 4. Mathematical notation density (more math symbols often indicates formal reasoning)
        math_symbols = sum(1 for c in thinking if c in "+-*/=^()[]{}∫∑∏√")
        math_factor = min(0.3, math_symbols / 100)
        
        # Combine confidence factors
        confidence_score = 0.3 + (0.7 * (length_factor + verification_factor + structure_factor + math_factor) / 4)
        confidence_scores.append(confidence_score)
    
    # Calculate answer consistency component
    answer_counts = {}
    for s in valid_solutions:
        answer_str = str(s['answer'])
        answer_counts[answer_str] = answer_counts.get(answer_str, 0) + 1
    
    consistency_scores = []
    for s in valid_solutions:
        answer_str = str(s['answer'])
        # Solutions with more common answers get higher consistency scores
        consistency_scores.append(answer_counts[answer_str] / len(valid_solutions))
    
    # Combine all components into final weights
    # Weight formula: 0.5*similarity + 0.3*confidence + 0.2*consistency
    combined_weights = []
    for i in range(len(valid_solutions)):
        weight = (0.5 * similarity_scores[i]) + (0.3 * confidence_scores[i]) + (0.2 * consistency_scores[i])
        combined_weights.append(weight)
    
    # Normalize weights to sum to 1
    total_weight = sum(combined_weights)
    if total_weight == 0:
        # Fallback to equal weights if all weights are zero
        weights = [1.0 / len(valid_solutions)] * len(valid_solutions)
    else:
        weights = [w / total_weight for w in combined_weights]
    
    # Count weighted votes for each answer
    answer_weights = {}
    for solution, weight in zip(valid_solutions, weights):
        answer_str = str(solution['answer'])
        if answer_str in answer_weights:
            answer_weights[answer_str] += weight
        else:
            answer_weights[answer_str] = weight
    
    # Find the answer with the highest weight
    if not answer_weights:
        return None, 0.0
    
    majority_answer, confidence = max(answer_weights.items(), key=lambda x: x[1])
    return majority_answer, confidence

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification and similarity-weighted majority voting"""
    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example or (('solution' not in example) and ('answer' not in example)):
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None
        # Extract the correct answer
        correct_answer = None
        if 'answer' in example and example['answer']:
            correct_answer = example['answer']
        else:
            correct_answer = extract_answer_from_solution(example['solution'])
        
        if correct_answer is None:
            logger.append(f"❌ Warning: Could not extract answer from solution for example {str(running_id)}")
            logger.print()
            return None

        main = get_model(config, role="main")
        solution_agent = FullSolutionAgent(main)
        solutions = []
        correct_count = 0
        best_solution = None
        
        for attempt in range(config.best_of):
            try:
                prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
                # Create numeric verifier
                verifier = NumericVerifier(tolerance=config.tolerance)
                is_correct, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                # Always append the solution, regardless of correctness
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'is_correct': is_correct,
                    'thinking': extract_thinking_section(current_solution) or current_solution
                })
                
                # Update statistics if correct
                if is_correct:
                    correct_count += 1
                    if best_solution is None:
                        best_solution = current_solution
            except Exception as e:
                logger.append(f"❌ Error in attempt {str(attempt + 1)} for example {str(running_id)}:")
                logger.append(f"Exception type: {type(e).__name__}")
                logger.append(f"Exception message: {str(e)}")
                import traceback
                logger.append(f"Traceback:\n{traceback.format_exc()}")
                
                # Retry this attempt up to 3 times
                for retry in range(3):
                    try:
                        logger.append(f"Retrying attempt {attempt + 1} (retry {retry + 1}/3)...")
                        current_solution = await solution_agent.generate(example["problem"])
                        
                        # Create numeric verifier
                        verifier = NumericVerifier(tolerance=config.tolerance)
                        is_correct, current_answer = await verifier.verify(
                            current_solution,
                            correct_answer,
                            example["problem"]
                        )
                        
                        if is_correct:
                            correct_count += 1
                            if best_solution is None:
                                best_solution = current_solution
                                
                        solutions.append({
                            'solution': current_solution,
                            'answer': current_answer,
                            'is_correct': is_correct,
                            'thinking': extract_thinking_section(current_solution) or current_solution
                        })
                        break  # Success, exit retry loop
                        
                    except Exception as retry_e:
                        logger.append(f"Retry {retry + 1} failed: {str(retry_e)}")
                        if retry == 2:  # Last retry failed
                            solution_info = {
                                'solution': f"Error occurred after 3 retries: {type(e).__name__} - {str(e)}",
                                'answer': None,
                                'is_correct': False,
                                'thinking': ""
                            }
                            solutions.append(solution_info)
                continue  # Move to next attempt
        
        # Initialize similarity checker
        similarity_checker = SolutionSimilarityChecker(config)
        
        # Extract thinking sections for similarity comparison
        thinking_texts = [s['thinking'] for s in solutions]
        
        # Compute similarity matrix
        similarity_matrix = similarity_checker.compute_similarity_matrix(thinking_texts)
        
        # Calculate confidence-boosted similarity majority answer
        confidence_boosted_answer, confidence = calculate_confidence_boosted_similarity(solutions, similarity_matrix)
        is_confidence_boosted_correct = False
        
        # Check if the confidence-boosted answer is correct
        if confidence_boosted_answer is not None:
            verifier = NumericVerifier(tolerance=config.tolerance)
            for s in solutions:
                if s['answer'] is not None and str(s['answer']) == confidence_boosted_answer:
                    is_confidence_boosted_correct = s['is_correct']
                    break
        
        # Calculate initial majority answer (standard majority voting)
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        initial_majority_answer = None
        is_initial_majority_correct = False
        if model_answers:
            initial_majority_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_initial_majority_correct = any(str(s['answer']) == initial_majority_answer and s['is_correct'] for s in solutions)
        
        # After applying confidence boosting, the final majority answer is the confidence-boosted one
        final_majority_answer = confidence_boosted_answer
        is_final_majority_correct = is_confidence_boosted_correct

        # Calculate thinking length statistics
        thinking_lengths = [len(s['thinking']) for s in solutions]
        correct_thinking_lengths = [len(s['thinking']) for s in solutions if s['is_correct']]
        incorrect_thinking_lengths = [len(s['thinking']) for s in solutions if not s['is_correct']]
        
        avg_thinking_length = sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0
        avg_correct_thinking = sum(correct_thinking_lengths) / len(correct_thinking_lengths) if correct_thinking_lengths else 0
        avg_incorrect_thinking = sum(incorrect_thinking_lengths) / len(incorrect_thinking_lengths) if incorrect_thinking_lengths else 0
        
        # Calculate average similarity for each solution
        avg_similarities = []
        for i in range(len(solutions)):
            similarities = [similarity_matrix[i, j].item() for j in range(len(solutions)) if i != j]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            avg_similarities.append(avg_similarity)
        
        # Create similarity distribution visualization
        if avg_similarities:
            # Create a simple ASCII histogram
            correct_similarities = [sim for sim, s in zip(avg_similarities, solutions) if s['is_correct']]
            incorrect_similarities = [sim for sim, s in zip(avg_similarities, solutions) if not s['is_correct']]
            
            correct_sim_hist = create_ascii_histogram(correct_similarities, "Correct solutions similarity")
            incorrect_sim_hist = create_ascii_histogram(incorrect_similarities, "Incorrect solutions similarity")
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Model answers: {[s['answer'] for s in solutions]}")
        logger.append(f"├─ Average similarities: {[f'{sim:.3f}' for sim in avg_similarities]}")
        logger.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logger.append(f"├─ Correct solutions: {correct_count}/{config.best_of}")
        logger.append(f"├─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        logger.append(f"├─ Initial majority answer: {initial_majority_answer}")
        logger.append(f"├─ Initial majority correct? {'Yes' if is_initial_majority_correct else 'No'}")
        logger.append(f"├─ Confidence-boosted majority answer: {confidence_boosted_answer}")
        logger.append(f"├─ Confidence-boosted confidence: {confidence:.2f}")
        logger.append(f"├─ Confidence-boosted correct? {'Yes' if is_confidence_boosted_correct else 'No'}")
        logger.append(f"├─ Final majority answer: {final_majority_answer}")
        logger.append(f"├─ Final majority correct? {'Yes' if is_final_majority_correct else 'No'}")
        logger.append(f"├─ Avg thinking length: {avg_thinking_length:.1f} chars")
        logger.append(f"├─ Avg correct thinking length: {avg_correct_thinking:.1f} chars")
        logger.append(f"└─ Avg incorrect thinking length: {avg_incorrect_thinking:.1f} chars")
        
        # Add similarity distributions
        if avg_similarities:
            logger.append("\n📊 Similarity Distributions:")
            logger.append(correct_sim_hist)
            logger.append(incorrect_sim_hist)
            
        # Print similarity matrix in a readable format
        logger.append("\n📊 Similarity Matrix:")
        matrix_str = []
        for i in range(len(solutions)):
            row = [f"{similarity_matrix[i, j].item():.2f}" for j in range(len(solutions))]
            matrix_str.append(" ".join(row))
        logger.append("\n".join(matrix_str))
            
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create individual entries for each solution
        result_entries = []
        
        # Add individual solution entries
        for i, s in enumerate(solutions):
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_solution': example['solution'],
                'correct_answer': correct_answer,
                'model_solution': s['solution'],
                'model_answer': s['answer'],
                'is_correct': s['is_correct'],
                'avg_similarity': avg_similarities[i],
                'attempt_number': i + 1,
                'total_attempts': len(solutions)
            })
        
        # Add statistics entry with similarity-weighted information
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [s['is_correct'] for s in solutions],
            'is_initial_majority_correct': is_initial_majority_correct,
            'initial_majority_answer': initial_majority_answer,
            'is_confidence_boosted_correct': is_confidence_boosted_correct,
            'confidence_boosted_answer': confidence_boosted_answer,
            'confidence_boosted_confidence': confidence,
            'is_final_majority_correct': is_final_majority_correct,
            'final_majority_answer': final_majority_answer,
            'success_rate': (correct_count/config.best_of)*100,
            'total_solutions': len(solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(solutions) - correct_count,
            'avg_similarities': avg_similarities,
            'avg_thinking_length': avg_thinking_length,
            'avg_correct_thinking': avg_correct_thinking,
            'avg_incorrect_thinking': avg_incorrect_thinking,
            'all_solutions_correct': all(s['is_correct'] for s in solutions)
        })
        
        return result_entries
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'is_correct_list': [],
            'is_initial_majority_correct': None,
            'initial_majority_answer': None,
            'is_similarity_weighted_correct': None,
            'similarity_weighted_answer': None,
            'similarity_weighted_confidence': 0.0,
            'is_final_majority_correct': None,
            'final_majority_answer': None,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'avg_similarities': [],
            'avg_thinking_length': 0,
            'avg_correct_thinking': 0,
            'avg_incorrect_thinking': 0,
            'all_solutions_correct': None
        }]


def create_ascii_histogram(data: List[float], title: str) -> str:
    """Create a simple ASCII histogram for the given data"""
    if not data:
        return f"{title}:\n  No data available"
    
    # Create bins
    min_val = min(data) if data else 0
    max_val = max(data) if data else 0
    
    if min_val == max_val:
        return f"{title}:\n  All values are {min_val:.3f}"
    
    # Create 5 bins
    bin_width = (max_val - min_val) / 5
    bins = [min_val + i * bin_width for i in range(6)]
    
    # Count values in each bin
    hist = [0] * 5
    for val in data:
        for i in range(5):
            if bins[i] <= val < bins[i+1]:
                hist[i] += 1
                break
        # Handle the last bin edge case
        if val == bins[5]:
            hist[4] += 1
    
    # Create ASCII representation
    result = [f"{title} (n={len(data)}):\n"]
    max_count = max(hist) if hist else 0
    scale = min(40, max_count)  # Scale to fit in console
    
    for i in range(5):
        bin_label = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
        bar_length = int((hist[i] / max_count) * scale) if max_count > 0 else 0
        bar = "█" * bar_length
        result.append(f"  {bin_label.rjust(15)}: {bar} ({hist[i]})")
    
    return "\n".join(result)

async def main():
    """Main function for benchmarking mathematical problem solving with similarity-weighted majority."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with similarity-weighted majority')
    
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        logger.print()
