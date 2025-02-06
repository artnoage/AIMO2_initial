import os
import asyncio
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import get_model, extract_answer_from_solution
from utils.agents import CompletionAgent
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> List[Dict]:
    """Process a single example by attempting to complete its partial solution"""
    try:
        logger = BenchmarkLogger()
        
        if not isinstance(example, dict) or 'problem' not in example or 'partial_solution' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None

        # Skip if no partial solution
        if not example.get('partial_solution'):
            logger.append(f"❌ Warning: No partial solution found for example {running_id}")
            logger.print()
            return []

        # Initialize model and agent
        main = get_model(config, role="main")
        completion_agent = CompletionAgent(main)
        
        # Create logs list
        logs = []
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n📝 Partial Solution:")
        logs.append(f"{example['partial_solution'][:200]}...")
        
        # Try completions
        extension_possible = False
        correct_answer = example.get('correct_answer')
        
        for attempt in range(config.best_of):
            try:
                # Get completion
                completion = await completion_agent.generate(
                    example['problem'],
                    example['partial_solution']
                )
                complete_solution = example['partial_solution'] + completion
                
                # Extract and compare answers
                completed_answer = extract_answer_from_solution(complete_solution)
                if completed_answer == correct_answer:
                    extension_possible = True
                    logs.append(f"\n✓ Found correct completion on attempt {attempt + 1}")
                    break
                    
            except Exception as e:
                logs.append(f"❌ Error in completion attempt {attempt + 1}: {str(e)}")
                continue
        
        # Create result entries
        results = []
        
        # Add benchmark data - carry over tutor benchmark info and add completion results
        results.append({
            'id': example_id,
            'data_type': 'comp_ben',
            'problem': example['problem'],
            'solution': example['solution'],
            'expected_verdict': example['expected_verdict'],
            'tutor_verdicts': example['tutor_verdicts'],
            'tutor_analyses': example['tutor_analyses'],
            'tutor_substitutions': example['tutor_substitutions'],
            'verdict_matches': example['verdict_matches'],
            'partial_solution': example['partial_solution'],
            'extension_possible': extension_possible
        })
        
        # Add statistics
        results.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': [extension_possible],
            'is_most_common_correct': extension_possible,
            'success_rate': 100.0 if extension_possible else 0.0,
            'total_solutions': 1,
            'correct_solutions': 1 if extension_possible else 0,
            'incorrect_solutions': 0 if extension_possible else 1,
            'tournament_winner_correct': None,
            'judge_accuracy': None,
            'judge_decisions': 0,
            'all_solutions_correct': extension_possible
        })
        
        # Log results
        logs.append(f"\n📊 Extension possible: {'Yes' if extension_possible else 'No'}")
        
        # Print all logs
        for log in logs:
            logger.append(log)
        logger.print()
        
        return results

    except Exception as e:
        logger = BenchmarkLogger()
        logger.append(f"❌ Error processing example {running_id}: {str(e)}")
        logger.print()
        return []

async def main():
    """Main function for completion benchmarking approach"""
    config = BenchmarkConfig.from_args('Completion benchmarking approach')
    
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
