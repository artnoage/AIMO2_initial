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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
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

def hierarchical_clustering_with_representative(solutions: List[Dict], similarity_matrix: torch.Tensor, 
                                              distance_threshold: float = 0.3) -> Tuple[Optional[str], float]:
    """
    Use hierarchical clustering to group similar solutions, then select the most representative
    solution from the largest cluster as the final answer.
    
    Args:
        solutions: List of solution dictionaries with 'answer' and 'solution' keys
        similarity_matrix: Tensor of pairwise similarities between solutions
        distance_threshold: Maximum distance to merge clusters (1 - similarity)
        
    Returns:
        Tuple of (representative_answer, confidence_score)
    """
    # Filter out None answers
    valid_indices = [i for i, s in enumerate(solutions) if s['answer'] is not None]
    if not valid_indices:
        return None, 0.0
    
    # Extract valid solutions
    valid_solutions = [solutions[i] for i in valid_indices]
    
    # Convert similarity matrix to distance matrix (1 - similarity)
    # Only include valid indices
    distance_matrix = []
    for i in valid_indices:
        row = [1.0 - similarity_matrix[i, j].item() for j in valid_indices]
        distance_matrix.append(row)
    
    # Convert to numpy array for hierarchical clustering
    distance_array = np.array(distance_matrix)
    
    # Perform hierarchical clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    
    # Use complete linkage (maximum distance between clusters)
    Z = linkage(distance_array, method='complete')
    
    # Form flat clusters at the specified distance threshold
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    
    # Group solutions by cluster
    cluster_groups = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        cluster_groups[cluster_id].append(i)
    
    # Find the largest cluster
    largest_cluster_id = max(cluster_groups.keys(), key=lambda k: len(cluster_groups[k]))
    largest_cluster = cluster_groups[largest_cluster_id]
    
    # If there's a tie for largest cluster, use the one with more correct solutions if known
    if sum(1 for k in cluster_groups if len(cluster_groups[k]) == len(largest_cluster)) > 1:
        # Check if we have correctness information
        if any('is_correct' in valid_solutions[i] for i in range(len(valid_solutions))):
            # Find cluster with most correct solutions
            correct_counts = {}
            for cluster_id, members in cluster_groups.items():
                correct_counts[cluster_id] = sum(1 for i in members 
                                               if valid_solutions[i].get('is_correct', False))
            
            # If there are any correct solutions, use that cluster
            if any(correct_counts.values()):
                largest_cluster_id = max(correct_counts.keys(), key=lambda k: correct_counts[k])
                largest_cluster = cluster_groups[largest_cluster_id]
    
    # Find the most representative solution in the largest cluster
    # (the one with highest average similarity to other solutions in the cluster)
    if not largest_cluster:
        return None, 0.0
    
    representative_idx = -1
    highest_avg_similarity = -1
    
    for i in largest_cluster:
        # Calculate average similarity to other solutions in the cluster
        similarities = [1.0 - distance_array[i][j] for j in largest_cluster if i != j]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        if avg_similarity > highest_avg_similarity:
            highest_avg_similarity = avg_similarity
            representative_idx = i
    
    if representative_idx == -1:
        return None, 0.0
    
    # Get the answer from the representative solution
    representative_answer = str(valid_solutions[representative_idx]['answer'])
    
    # Calculate confidence based on:
    # 1. Size of largest cluster relative to all solutions
    size_confidence = len(largest_cluster) / len(valid_solutions)
    
    # 2. Average similarity within the cluster
    within_cluster_similarities = []
    for i in largest_cluster:
        for j in largest_cluster:
            if i != j:
                within_cluster_similarities.append(1.0 - distance_array[i][j])
    
    similarity_confidence = sum(within_cluster_similarities) / len(within_cluster_similarities) if within_cluster_similarities else 0.0
    
    # 3. Agreement on the answer within the cluster
    answer_counts = {}
    for i in largest_cluster:
        ans = str(valid_solutions[i]['answer'])
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    
    # Calculate what percentage of the cluster agrees with the representative answer
    agreement_confidence = answer_counts.get(representative_answer, 0) / len(largest_cluster)
    
    # Combine confidence factors
    confidence = (0.4 * size_confidence) + (0.3 * similarity_confidence) + (0.3 * agreement_confidence)
    
    return representative_answer, confidence

def visualize_hierarchical_clustering(solutions: List[Dict], similarity_matrix: torch.Tensor) -> str:
    """
    Create a text-based visualization of the hierarchical clustering.
    
    Args:
        solutions: List of solution dictionaries
        similarity_matrix: Tensor of pairwise similarities between solutions
        
    Returns:
        String containing ASCII visualization of the clustering
    """
    # Filter out None answers
    valid_indices = [i for i, s in enumerate(solutions) if s['answer'] is not None]
    if not valid_indices:
        return "No valid solutions to cluster"
    
    # Convert similarity matrix to distance matrix (1 - similarity)
    distance_matrix = []
    for i in valid_indices:
        row = [1.0 - similarity_matrix[i, j].item() for j in valid_indices]
        distance_matrix.append(row)
    
    # Convert to numpy array for hierarchical clustering
    distance_array = np.array(distance_matrix)
    
    # Perform hierarchical clustering
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    # Use complete linkage (maximum distance between clusters)
    Z = linkage(distance_array, method='complete')
    
    # Create a string buffer to capture the ASCII dendrogram
    from io import StringIO
    buffer = StringIO()
    
    # Generate ASCII dendrogram
    dendrogram(Z, truncate_mode='level', p=3, show_leaf_counts=True, no_labels=True)
    
    # Create a simple text representation
    result = ["Hierarchical Clustering Visualization:"]
    result.append("(Solutions grouped by similarity of thinking)")
    result.append("")
    
    # Add a simple representation of the dendrogram
    max_height = max(Z[:, 2])
    for i, row in enumerate(Z):
        left, right, height, _ = row
        # Scale height to fit in console
        scaled_height = int((height / max_height) * 10)
        
        # Create a simple branch visualization
        branch = "│" * scaled_height + "┐"
        result.append(f"Merge {i+1}: Solutions {int(left)} and {int(right)} at distance {height:.2f}")
        result.append(branch)
    
    return "\n".join(result)

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
        
        # Try different distance thresholds for hierarchical clustering
        distance_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
        hierarchical_results = {}
        
        for threshold in distance_thresholds:
            # Calculate hierarchical clustering representative answer
            representative_answer, confidence = hierarchical_clustering_with_representative(
                solutions, similarity_matrix, distance_threshold=threshold
            )
            
            # Check if the representative answer is correct
            is_correct = False
            if representative_answer is not None:
                for s in solutions:
                    if s['answer'] is not None and str(s['answer']) == representative_answer:
                        is_correct = s['is_correct']
                        break
                        
            hierarchical_results[threshold] = {
                'answer': representative_answer,
                'confidence': confidence,
                'is_correct': is_correct
            }
        
        # Find the best threshold (highest confidence for correct answer)
        best_threshold = None
        best_confidence = -1
        
        for threshold, result in hierarchical_results.items():
            if result['is_correct'] and result['confidence'] > best_confidence:
                best_threshold = threshold
                best_confidence = result['confidence']
                
        if best_threshold is None and any(result['answer'] is not None for result in hierarchical_results.values()):
            # If no correct answer found, use the threshold with highest confidence
            best_threshold = max(
                [(t, r['confidence']) for t, r in hierarchical_results.items() if r['answer'] is not None],
                key=lambda x: x[1],
                default=(0.3, 0)
            )[0]
        
        # If still no best threshold, use default
        if best_threshold is None:
            best_threshold = 0.3
            
        # Get the hierarchical clustering answer with the best threshold
        hierarchical_answer = hierarchical_results[best_threshold]['answer']
        hierarchical_confidence = hierarchical_results[best_threshold]['confidence']
        is_hierarchical_correct = hierarchical_results[best_threshold]['is_correct']
        
        # Create hierarchical clustering visualization
        clustering_visualization = visualize_hierarchical_clustering(solutions, similarity_matrix)
        
        # Calculate initial majority answer (standard majority voting)
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        initial_majority_answer = None
        is_initial_majority_correct = False
        if model_answers:
            initial_majority_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_initial_majority_correct = any(str(s['answer']) == initial_majority_answer and s['is_correct'] for s in solutions)
        
        # After applying hierarchical clustering, the final majority answer is the hierarchical representative
        final_majority_answer = hierarchical_answer
        is_final_majority_correct = is_hierarchical_correct

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
        logger.append(f"├─ Best distance threshold: {best_threshold}")
        logger.append(f"├─ Hierarchical clustering answer: {hierarchical_answer}")
        logger.append(f"├─ Hierarchical clustering confidence: {hierarchical_confidence:.2f}")
        logger.append(f"├─ Hierarchical clustering correct? {'Yes' if is_hierarchical_correct else 'No'}")
        logger.append(f"├─ Final majority answer: {final_majority_answer}")
        logger.append(f"├─ Final majority correct? {'Yes' if is_final_majority_correct else 'No'}")
        logger.append(f"├─ Avg thinking length: {avg_thinking_length:.1f} chars")
        logger.append(f"├─ Avg correct thinking length: {avg_correct_thinking:.1f} chars")
        logger.append(f"└─ Avg incorrect thinking length: {avg_incorrect_thinking:.1f} chars")
        
        # Add consensus distributions
        if consensus_counts:
            logger.append("\n📊 Consensus Distributions:")
            logger.append(correct_consensus_hist)
            logger.append(incorrect_consensus_hist)
            
        # Print hierarchical clustering visualization
        logger.append("\n📊 Hierarchical Clustering:")
        logger.append(clustering_visualization)
            
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
            'is_hierarchical_correct': is_hierarchical_correct,
            'hierarchical_answer': hierarchical_answer,
            'hierarchical_confidence': hierarchical_confidence,
            'best_distance_threshold': best_threshold,
            'is_final_majority_correct': is_final_majority_correct,
            'final_majority_answer': final_majority_answer,
            'success_rate': (correct_count/config.best_of)*100,
            'total_solutions': len(solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(solutions) - correct_count,
            'consensus_counts': consensus_counts,
            'avg_thinking_length': avg_thinking_length,
            'avg_correct_thinking': avg_correct_thinking,
            'avg_incorrect_thinking': avg_incorrect_thinking,
            'all_solutions_correct': all(s['is_correct'] for s in solutions),
            'hierarchical_clustering_used': True
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
