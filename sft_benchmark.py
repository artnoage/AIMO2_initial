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
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a text string"""
    return len(tokenizer(text)['input_ids'])

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> Optional[Dict]:
    """Process a single example with configured verification"""
    logger = BenchmarkLogger()
    try:
        if not isinstance(example, dict) or 'problem' not in example or (('solution' not in example) and ('answer' not in example)):
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None
            
        correct_answer = None
        if correct_answer==None:
            correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            logger.append(f"❌ Warning: Could not extract answer from solution for example {str(running_id)}")
            logger.print()
            return None

        # Initialize tokenizer for length checking
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mathstral-7B-v0.1")

        main = get_model(config, role="main")
        solution_agent = FullSolutionAgent(main)
        solutions = []
        correct_solutions = []
        correct_count = 0
        
        for attempt in range(config.best_of):
            try:
                prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
                
                # Check solution length
                if count_tokens(current_solution, tokenizer) > 1000:
                    continue
                
                # Check prompt length
                if count_tokens(prompt, tokenizer) > 1000:
                    continue
                
                # Create numeric verifier
                verifier = NumericVerifier(tolerance=config.tolerance)
                is_correct, current_answer = await verifier.verify(
                    current_solution,
                    correct_answer,
                    example["problem"]
                )
                
                # Only append if both solution and prompt are within token limits
                solutions.append({
                    'solution': current_solution,
                    'answer': current_answer,
                    'is_correct': is_correct,
                    'prompt': prompt
                })
                
                # Store correct solutions separately
                if is_correct:
                    correct_count += 1
                    correct_solutions.append({
                        'solution': current_solution,
                        'answer': current_answer,
                        'prompt': prompt
                    })
                    
            except Exception as e:
                logger.append(f"❌ Error in attempt {str(attempt + 1)} for example {str(running_id)}")
                continue

        # Calculate most common answer statistics
        model_answers = [s['answer'] for s in solutions if s['answer'] is not None]
        most_common_answer = None
        is_most_common_correct = False
        if model_answers:
            from collections import Counter
            most_common_answer = Counter(str(ans) for ans in model_answers).most_common(1)[0][0]
            is_most_common_correct = any(str(s['answer']) == most_common_answer and s['is_correct'] for s in solutions)

        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Model answers: {[s['answer'] for s in solutions]}")
        logger.append(f"├─ Correct/incorrect: {[1 if s['is_correct'] and s['answer'] is not None else 0 for s in solutions]}")
        logger.append(f"├─ Correct solutions: {correct_count}/{config.best_of}")
        logger.append(f"├─ Success rate: {(correct_count/config.best_of)*100:.1f}%")
        logger.append(f"├─ Most common answer: {most_common_answer}")
        logger.append(f"└─ Most common answer correct? {'Yes' if is_most_common_correct else 'No'}")
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()

        # Only return results if we have correct solutions
        if correct_solutions:
            # Get the first correct solution for the chat format
            first_correct = correct_solutions[0]
            return [
                {
                    'id': example_id,
                    'data_type': 'training',
                    'messages': [
                        {"role": "user", "content": first_correct['prompt']},
                        {"role": "assistant", "content": first_correct['solution']}
                    ]
                },
                {
                    'id': example_id,
                    'data_type': 'statistics',
                    'example_processed_successfully': True,
                    'is_correct_list': [s['is_correct'] for s in solutions],
                    'is_most_common_correct': is_most_common_correct,
                    'success_rate': (correct_count/config.best_of)*100,
                    'total_solutions': len(solutions),
                    'correct_solutions': correct_count,
                    'incorrect_solutions': len(solutions) - correct_count,
                    'tournament_winner_correct': None,
                    'judge_accuracy': None,
                    'judge_decisions': 0,
                    'all_solutions_correct': all(s['is_correct'] for s in solutions)
                }
            ]
        else:
            # Return only statistics if no correct solutions
            return [{
                'id': example_id,
                'data_type': 'statistics',
                'example_processed_successfully': True,
                'is_correct_list': [s['is_correct'] for s in solutions],
                'is_most_common_correct': is_most_common_correct,
                'success_rate': (correct_count/config.best_of)*100,
                'total_solutions': len(solutions),
                'correct_solutions': correct_count,
                'incorrect_solutions': len(solutions) - correct_count,
                'tournament_winner_correct': None,
                'judge_accuracy': None,
                'judge_decisions': 0,
                'all_solutions_correct': all(s['is_correct'] for s in solutions)
            }]
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'is_correct_list': [],
            'is_most_common_correct': None,
            'success_rate': 0,
            'total_solutions': 0,
            'correct_solutions': 0,
            'incorrect_solutions': 0,
            'tournament_winner_correct': None,
            'judge_accuracy': None,
            'judge_decisions': 0,
            'all_solutions_correct': None
        }]

async def main():
    """Main function for collecting SFT training data."""
    config = BenchmarkConfig.from_args('Collect SFT training data from mathematical problems')
    
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
