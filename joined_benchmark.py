import os
import asyncio
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import *
from utils.agents import *
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example using both main and auxiliary models with best-of sampling"""
    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example or (('solution' not in example) and ('answer' not in example)):
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            logger.append(f"❌ Warning: Could not extract answer from solution for example {str(running_id)}")
            logger.print()
            return None

        # Initialize models and agents
        main_model = get_model(config, role="main")
        aux_model = get_model(config, role="auxiliary")
        
        main_agent = FullSolutionAgent(main_model)
        aux_agent = FullSolutionAgent(aux_model)
        
        # Create numeric verifier
        verifier = NumericVerifier(tolerance=config.tolerance)
        
        # Store all solutions from both models
        all_solutions = []
        correct_count = 0
        
        # Generate best_of solutions for each model
        for attempt in range(config.best_of):
            try:
                # Get solutions from both agents
                main_prompt, main_solution = await main_agent.generate(example["problem"], return_prompt=True)
                aux_prompt, aux_solution = await aux_agent.generate(example["problem"], return_prompt=True)
                
                # Verify both solutions
                main_correct, main_answer = await verifier.verify(main_solution, correct_answer, example["problem"])
                aux_correct, aux_answer = await verifier.verify(aux_solution, correct_answer, example["problem"])
                
                # Add to solutions list with cross-reference of other model's correctness
                all_solutions.append({
                    'solution': main_solution, 
                    'answer': main_answer, 
                    'is_correct': main_correct, 
                    'agent': 'main',
                    'other_agent_correct': aux_correct,
                    'attempt_number': attempt + 1
                })
                
                all_solutions.append({
                    'solution': aux_solution, 
                    'answer': aux_answer, 
                    'is_correct': aux_correct, 
                    'agent': 'auxiliary',
                    'other_agent_correct': main_correct,
                    'attempt_number': attempt + 1
                })
                
                # Update correct count
                if main_correct:
                    correct_count += 1
                if aux_correct:
                    correct_count += 1
                    
            except Exception as e:
                logger.append(f"❌ Error in attempt {attempt + 1}: {str(e)}")
                # Continue with next attempt
        
        # If no solutions were generated successfully, return error
        if not all_solutions:
            logger.append(f"❌ Failed to generate any solutions for example {str(running_id)}")
            logger.print()
            return None
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        
        # Group solutions by attempt for logging
        for attempt in range(config.best_of):
            attempt_num = attempt + 1
            main_solutions = [s for s in all_solutions if s['agent'] == 'main' and s['attempt_number'] == attempt_num]
            aux_solutions = [s for s in all_solutions if s['agent'] == 'auxiliary' and s['attempt_number'] == attempt_num]
            
            if main_solutions and aux_solutions:
                main_sol = main_solutions[0]
                aux_sol = aux_solutions[0]
                logger.append(f"\nAttempt {attempt_num}:")
                logger.append(f"├─ Main model answer: {main_sol['answer']} (Correct: {main_sol['is_correct']})")
                logger.append(f"└─ Auxiliary model answer: {aux_sol['answer']} (Correct: {aux_sol['is_correct']})")
        
        # Calculate most common answer statistics for each model
        from collections import Counter
        
        # For main model
        main_solutions = [s for s in all_solutions if s['agent'] == 'main']
        aux_solutions = [s for s in all_solutions if s['agent'] == 'auxiliary']
        
        main_answers = [str(s['answer']) for s in main_solutions if s['answer'] is not None]
        main_most_common_answer = Counter(main_answers).most_common(1)[0][0] if main_answers else None
        main_most_common_correct = any(str(s['answer']) == main_most_common_answer and s['is_correct'] 
                                      for s in main_solutions) if main_most_common_answer else False
        
        # For auxiliary model
        aux_answers = [str(s['answer']) for s in aux_solutions if s['answer'] is not None]
        aux_most_common_answer = Counter(aux_answers).most_common(1)[0][0] if aux_answers else None
        aux_most_common_correct = any(str(s['answer']) == aux_most_common_answer and s['is_correct'] 
                                     for s in aux_solutions) if aux_most_common_answer else False
        
        # For combined models (all answers from both models)
        combined_answers = [str(s['answer']) for s in all_solutions if s['answer'] is not None]
        combined_most_common_answer = Counter(combined_answers).most_common(1)[0][0] if combined_answers else None
        combined_most_common_correct = any(str(s['answer']) == combined_most_common_answer and s['is_correct'] 
                                          for s in all_solutions) if combined_most_common_answer else False
        
        # Overall statistics
        total_attempts = config.best_of * 2  # 2 models per attempt
        logger.append(f"\nOverall Statistics:")
        logger.append(f"├─ Total correct: {correct_count}/{total_attempts}")
        logger.append(f"├─ Main model most common answer: {main_most_common_answer} (Correct: {main_most_common_correct})")
        logger.append(f"├─ Auxiliary model most common answer: {aux_most_common_answer} (Correct: {aux_most_common_correct})")
        logger.append(f"└─ Combined models most common answer: {combined_most_common_answer} (Correct: {combined_most_common_correct})")
        logger.append("="*80)
        
        results = []
        
        # Add individual entries for each solution
        for solution in all_solutions:
            # Get the appropriate prompt
            prompt = example["problem"]
            
            results.append({
                'id': example_id,
                'data_type': 'training',
                'problem': example['problem'],
                'correct_solution': example['solution'],
                'correct_answer': correct_answer,
                'model_solution': solution['solution'],
                'model_answer': solution['answer'],
                'is_correct': solution['is_correct'],
                'agent_type': solution['agent'],
                'other_agent_correct': solution['other_agent_correct'],
                'attempt_number': solution['attempt_number'],
                'total_attempts': config.best_of
            })
        
        # Calculate detailed statistics for joined benchmark
        main_solutions = [s for s in all_solutions if s['agent'] == 'main']
        aux_solutions = [s for s in all_solutions if s['agent'] == 'auxiliary']
        
        main_correct_count = sum(1 for s in main_solutions if s['is_correct'])
        aux_correct_count = sum(1 for s in aux_solutions if s['is_correct'])
        
        # Calculate agreement statistics
        both_correct_count = 0
        both_wrong_count = 0
        disagreement_count = 0
        
        for i in range(config.best_of):
            main_sol = next((s for s in main_solutions if s['attempt_number'] == i+1), None)
            aux_sol = next((s for s in aux_solutions if s['attempt_number'] == i+1), None)
            
            if main_sol and aux_sol:
                if main_sol['is_correct'] and aux_sol['is_correct']:
                    both_correct_count += 1
                elif not main_sol['is_correct'] and not aux_sol['is_correct']:
                    both_wrong_count += 1
                else:
                    disagreement_count += 1
        
        # Calculate which model performs better when they disagree
        main_better_when_disagree = 0
        aux_better_when_disagree = 0
        
        for i in range(config.best_of):
            main_sol = next((s for s in main_solutions if s['attempt_number'] == i+1), None)
            aux_sol = next((s for s in aux_solutions if s['attempt_number'] == i+1), None)
            
            if main_sol and aux_sol and main_sol['is_correct'] != aux_sol['is_correct']:
                if main_sol['is_correct']:
                    main_better_when_disagree += 1
                else:
                    aux_better_when_disagree += 1
        
        # Always add detailed statistics
        results.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [s['is_correct'] for s in all_solutions],
            'is_most_common_correct': sum(1 for s in all_solutions if s['is_correct']) > len(all_solutions)/2,
            'success_rate': (correct_count/len(all_solutions))*100,
            'total_solutions': len(all_solutions),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(all_solutions) - correct_count,
            'judge_accuracy': None,
            'judge_decisions': 0,
            'all_solutions_correct': all(s['is_correct'] for s in all_solutions),
            'main_model_correct_count': main_correct_count,
            'aux_model_correct_count': aux_correct_count,
            'total_attempts_per_model': config.best_of,  # This is per example
            
            # Add key joined benchmark statistics
            'both_correct_count': both_correct_count,
            'both_wrong_count': both_wrong_count,
            'disagreement_count': disagreement_count,
            'main_better_when_disagree': main_better_when_disagree,
            'aux_better_when_disagree': aux_better_when_disagree,
            'agreement_rate': ((both_correct_count + both_wrong_count) / config.best_of) * 100,
            'main_success_rate': (main_correct_count / config.best_of) * 100,
            'aux_success_rate': (aux_correct_count / config.best_of) * 100,
            'performance_gap': ((main_correct_count - aux_correct_count) / config.best_of) * 100,
            
            # Add most common answer statistics
            'main_most_common_answer': main_most_common_answer,
            'main_most_common_correct': main_most_common_correct,
            'aux_most_common_answer': aux_most_common_answer,
            'aux_most_common_correct': aux_most_common_correct,
            'combined_most_common_answer': combined_most_common_answer,
            'combined_most_common_correct': combined_most_common_correct
        })
        
        # Print logs before returning results
        logger.print()
        
        # Return results with logs
        return {
            'results': results,
            'logs': '\n'.join(logger.logs) if logger.logs else ""
        }
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        # Print logs before returning results
        logger.print()
        
        # Return results with logs
        return {
            'results': [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': False,
                'is_correct_list': [],
                'is_most_common_correct': None,
                'success_rate': 0,
                'total_solutions': 0,
                'correct_solutions': 0,
                'incorrect_solutions': 0,
                'judge_accuracy': None,
                'judge_decisions': 0,
                'all_solutions_correct': None
            }],
            'logs': '\n'.join(logger.logs) if logger.logs else ""
        }

async def main():
    """Main function for benchmarking mathematical problem solving with both main and auxiliary models."""
    config = BenchmarkConfig.from_args('Benchmark main and auxiliary models on mathematical problems')
    
    # Run the benchmark
    tracker = ProgressTracker(total_examples=0, config=config)
    await tracker.run_benchmark(process_example_func=process_example)
    
    # The ProgressTracker will handle printing the final statistics

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Benchmark interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        logger.append(f"Traceback:\n{traceback.format_exc()}")
        logger.print()
