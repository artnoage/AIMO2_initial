import os
import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
import sys
import torch
from collections import Counter, defaultdict
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

def calculate_consensus_weighted_majority(solutions: List[Dict], similarity_matrix: torch.Tensor, 
                                         threshold: float = 0.7) -> Tuple[Optional[str], float]:
    """
    Calculate the majority answer weighted by consensus similarity.
    Solutions that are similar to more other solutions (above threshold) get more weight.
    
    Args:
        solutions: List of solution dictionaries with 'answer' and 'solution' keys
        similarity_matrix: Tensor of pairwise similarities between solutions
        threshold: Similarity threshold to consider two solutions as "in agreement"
        
    Returns:
        Tuple of (majority_answer, confidence_score)
    """
    # Filter out None answers
    valid_indices = [i for i, s in enumerate(solutions) if s['answer'] is not None]
    if not valid_indices:
        return None, 0.0
    
    # Extract valid solutions
    valid_solutions = [solutions[i] for i in valid_indices]
    
    # Calculate consensus count for each solution
    # For each solution, count how many other solutions it has high similarity with
    consensus_counts = []
    for i in valid_indices:
        # Count solutions with similarity above threshold
        similar_count = sum(1 for j in valid_indices if i != j and similarity_matrix[i, j].item() >= threshold)
        consensus_counts.append(similar_count)
    
    # Use consensus counts as weights
    weights = consensus_counts
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        # Fallback to equal weights if all weights are zero
        weights = [1.0 / len(valid_solutions)] * len(valid_solutions)
    else:
        weights = [w / total_weight for w in weights]
    
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

def find_solution_clusters(solutions: List[Dict], similarity_matrix: torch.Tensor, 
                          threshold: float = 0.7) -> List[List[int]]:
    """
    Group solutions into clusters based on similarity.
    
    Args:
        solutions: List of solution dictionaries
        similarity_matrix: Tensor of pairwise similarities between solutions
        threshold: Similarity threshold to consider two solutions as part of the same cluster
        
    Returns:
        List of clusters, where each cluster is a list of solution indices
    """
    n = len(solutions)
    # Track which solutions have been assigned to clusters
    assigned = [False] * n
    clusters = []
    
    # Process each unassigned solution
    for i in range(n):
        if assigned[i]:
            continue
            
        # Start a new cluster with this solution
        cluster = [i]
        assigned[i] = True
        
        # Find all similar solutions
        for j in range(n):
            if not assigned[j] and i != j and similarity_matrix[i, j].item() >= threshold:
                cluster.append(j)
                assigned[j] = True
                
        clusters.append(cluster)
    
    # Sort clusters by size (largest first)
    return sorted(clusters, key=len, reverse=True)

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification and consensus-weighted majority voting"""
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
        
        # Try different threshold values for consensus
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        consensus_results = {}
        
        for threshold in thresholds:
            # Calculate consensus-weighted majority answer
            consensus_answer, confidence = calculate_consensus_weighted_majority(
                solutions, similarity_matrix, threshold=threshold
            )
            
            # Check if the consensus-weighted answer is correct
            is_correct = False
            if consensus_answer is not None:
                for s in solutions:
                    if s['answer'] is not None and str(s['answer']) == consensus_answer:
                        is_correct = s['is_correct']
                        break
                        
            consensus_results[threshold] = {
                'answer': consensus_answer,
                'confidence': confidence,
                'is_correct': is_correct
            }
        
        # Find the best threshold (highest confidence for correct answer)
        best_threshold = None
        best_confidence = -1
        
        for threshold, result in consensus_results.items():
            if result['is_correct'] and result['confidence'] > best_confidence:
                best_threshold = threshold
                best_confidence = result['confidence']
                
        if best_threshold is None and any(result['answer'] is not None for result in consensus_results.values()):
            # If no correct answer found, use the threshold with highest confidence
            best_threshold = max(
                [(t, r['confidence']) for t, r in consensus_results.items() if r['answer'] is not None],
                key=lambda x: x[1],
                default=(0.7, 0)
            )[0]
        
        # If still no best threshold, use default
        if best_threshold is None:
            best_threshold = 0.7
            
        # Get the consensus answer with the best threshold
        consensus_weighted_answer = consensus_results[best_threshold]['answer']
        consensus_confidence = consensus_results[best_threshold]['confidence']
        is_consensus_weighted_correct = consensus_results[best_threshold]['is_correct']
        
        # Find solution clusters
        clusters = find_solution_clusters(solutions, similarity_matrix, threshold=best_threshold)
        
        # Calculate standard majority answer for comparison
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        most_common_answer = None
        is_most_common_correct = False
        if model_answers:
            most_common_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_most_common_correct = any(str(s['answer']) == most_common_answer and s['is_correct'] for s in solutions)
            
        # Store the initial majority answer for comparison
        initial_majority_answer = most_common_answer
        is_initial_majority_correct = is_most_common_correct

        # Calculate thinking length statistics
        thinking_lengths = [len(s['thinking']) for s in solutions]
        correct_thinking_lengths = [len(s['thinking']) for s in solutions if s['is_correct']]
        incorrect_thinking_lengths = [len(s['thinking']) for s in solutions if not s['is_correct']]
        
        avg_thinking_length = sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0
        avg_correct_thinking = sum(correct_thinking_lengths) / len(correct_thinking_lengths) if correct_thinking_lengths else 0
        avg_incorrect_thinking = sum(incorrect_thinking_lengths) / len(incorrect_thinking_lengths) if incorrect_thinking_lengths else 0
        
        # Calculate consensus count for each solution
        consensus_counts = []
        for i in range(len(solutions)):
            # Count solutions with similarity above threshold
            similar_count = sum(1 for j in range(len(solutions)) if i != j and similarity_matrix[i, j].item() >= best_threshold)
            consensus_counts.append(similar_count)
        
        # Create consensus distribution visualization
        if consensus_counts:
            # Create a simple ASCII histogram
            correct_consensus = [count for count, s in zip(consensus_counts, solutions) if s['is_correct']]
            incorrect_consensus = [count for count, s in zip(consensus_counts, solutions) if not s['is_correct']]
            
            correct_consensus_hist = create_ascii_histogram(correct_consensus, "Correct solutions consensus count")
            incorrect_consensus_hist = create_ascii_histogram(incorrect_consensus, "Incorrect solutions consensus count")
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Model answers: {[s['answer'] for s in solutions]}")
        logger.append(f"├─ Consensus counts: {consensus_counts}")
        logger.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logger.append(f"├─ Correct solutions: {correct_count}/{config.best_of}")
        logger.append(f"├─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        logger.append(f"├─ Initial majority answer: {initial_majority_answer}")
        logger.append(f"├─ Initial majority correct? {'Yes' if is_initial_majority_correct else 'No'}")
        logger.append(f"├─ Best threshold: {best_threshold}")
        logger.append(f"├─ Consensus-weighted majority answer: {consensus_weighted_answer}")
        logger.append(f"├─ Consensus-weighted confidence: {consensus_confidence:.2f}")
        logger.append(f"├─ Consensus-weighted correct? {'Yes' if is_consensus_weighted_correct else 'No'}")
        logger.append(f"├─ Final majority answer: {most_common_answer}")
        logger.append(f"├─ Final majority correct? {'Yes' if is_most_common_correct else 'No'}")
        logger.append(f"├─ Avg thinking length: {avg_thinking_length:.1f} chars")
        logger.append(f"├─ Avg correct thinking length: {avg_correct_thinking:.1f} chars")
        logger.append(f"└─ Avg incorrect thinking length: {avg_incorrect_thinking:.1f} chars")
        
        # Add consensus distributions
        if consensus_counts:
            logger.append("\n📊 Consensus Distributions:")
            logger.append(correct_consensus_hist)
            logger.append(incorrect_consensus_hist)
            
        # Print solution clusters
        logger.append("\n📊 Solution Clusters:")
        for i, cluster in enumerate(clusters):
            cluster_answers = [str(solutions[idx]['answer']) for idx in cluster if solutions[idx]['answer'] is not None]
            most_common = Counter(cluster_answers).most_common(1)[0][0] if cluster_answers else "None"
            correct = any(solutions[idx]['is_correct'] for idx in cluster)
            logger.append(f"Cluster {i+1} (size: {len(cluster)}, answer: {most_common}, correct: {'Yes' if correct else 'No'})")
            
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
                'consensus_count': consensus_counts[i],
                'attempt_number': i + 1,
                'total_attempts': len(solutions)
            })
        
        # Add statistics entry with consensus-weighted information
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [s['is_correct'] for s in solutions],
            'is_initial_majority_correct': is_initial_majority_correct,
            'initial_majority_answer': initial_majority_answer,
            'is_consensus_weighted_correct': is_consensus_weighted_correct,
            'consensus_weighted_answer': consensus_weighted_answer,
            'consensus_weighted_confidence': consensus_confidence,
            'best_threshold': best_threshold,
            'is_final_majority_correct': is_most_common_correct,
            'final_majority_answer': most_common_answer,
            'success_rate': (correct_count/config.best_of)*100,
            'total_solutions': len(solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(solutions) - correct_count,
            'consensus_counts': consensus_counts,
            'avg_thinking_length': avg_thinking_length,
            'avg_correct_thinking': avg_correct_thinking,
            'avg_incorrect_thinking': avg_incorrect_thinking,
            'all_solutions_correct': all(s['is_correct'] for s in solutions),
            'cluster_count': len(clusters),
            'largest_cluster_size': len(clusters[0]) if clusters else 0
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
            'is_consensus_weighted_correct': None,
            'consensus_weighted_answer': None,
            'consensus_weighted_confidence': 0.0,
            'best_threshold': None,
            'is_final_majority_correct': None,
            'final_majority_answer': None,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'consensus_counts': [],
            'avg_thinking_length': 0,
            'avg_correct_thinking': 0,
            'avg_incorrect_thinking': 0,
            'all_solutions_correct': None,
            'cluster_count': 0,
            'largest_cluster_size': 0
        }]


def create_ascii_histogram(data: List[int], title: str) -> str:
    """Create a simple ASCII histogram for the given data"""
    if not data:
        return f"{title}:\n  No data available"
    
    # Create bins
    min_val = min(data) if data else 0
    max_val = max(data) if data else 0
    
    if min_val == max_val:
        return f"{title}:\n  All values are {min_val}"
    
    # Create bins based on the range of values
    if max_val - min_val <= 5:
        # For small ranges, use integer bins
        bins = list(range(min_val, max_val + 1))
        bin_labels = [str(b) for b in bins[:-1]]
    else:
        # For larger ranges, use 5 bins
        bin_width = max(1, (max_val - min_val) // 5)
        bins = list(range(min_val, max_val + bin_width, bin_width))
        bin_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
    
    # Count values in each bin
    hist = [0] * (len(bins) - 1)
    for val in data:
        for i in range(len(bins) - 1):
            if bins[i] <= val < bins[i+1]:
                hist[i] += 1
                break
        # Handle the last bin edge case
        if val == bins[-1]:
            hist[-1] += 1
    
    # Create ASCII representation
    result = [f"{title} (n={len(data)}):\n"]
    max_count = max(hist) if hist else 0
    scale = min(40, max_count)  # Scale to fit in console
    
    for i in range(len(hist)):
        bin_label = bin_labels[i]
        bar_length = int((hist[i] / max_count) * scale) if max_count > 0 else 0
        bar = "█" * bar_length
        result.append(f"  {bin_label.rjust(10)}: {bar} ({hist[i]})")
    
    return "\n".join(result)

async def main():
    """Main function for benchmarking mathematical problem solving with consensus-weighted majority."""
    config = BenchmarkConfig.from_args('Benchmark model on mathematical problems with consensus-weighted majority')
    
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
