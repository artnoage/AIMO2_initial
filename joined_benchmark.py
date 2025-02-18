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
    """Process a single example using both main and auxiliary models"""
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
        
        # Get solutions from both agents
        main_prompt, main_solution = await main_agent.generate(example["problem"], return_prompt=True)
        aux_prompt, aux_solution = await aux_agent.generate(example["problem"], return_prompt=True)
        
        # Create numeric verifier
        verifier = NumericVerifier(tolerance=config.tolerance)
        
        # Verify both solutions
        main_correct, main_answer = await verifier.verify(main_solution, correct_answer, example["problem"])
        aux_correct, aux_answer = await verifier.verify(aux_solution, correct_answer, example["problem"])
        
        solutions = [
            {'solution': main_solution, 'answer': main_answer, 'is_correct': main_correct, 'agent': 'main'},
            {'solution': aux_solution, 'answer': aux_answer, 'is_correct': aux_correct, 'agent': 'auxiliary'}
        ]
        
        correct_count = sum(1 for s in solutions if s['is_correct'])
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Main model answer: {main_answer} (Correct: {main_correct})")
        logger.append(f"├─ Auxiliary model answer: {aux_answer} (Correct: {aux_correct})")
        logger.append(f"└─ Total correct: {correct_count}/2")
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        results = []
        
        # Add training data if any solution is correct
        if any(s['is_correct'] for s in solutions):
            # Get the first correct solution
            correct_solution = next(s for s in solutions if s['is_correct'])
            prompt = main_prompt if correct_solution['agent'] == 'main' else aux_prompt
            
            results.append({
                'id': example_id,
                'data_type': 'training',
                'messages': [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": correct_solution['solution']}
                ],
                'agent_type': correct_solution['agent']  # Tag which agent was correct
            })
            
        # Always add statistics
        results.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'main_correct': main_correct,
            'auxiliary_correct': aux_correct,
            'total_correct': correct_count,
            'success_rate': (correct_count/2)*100,
            'main_answer': main_answer,
            'auxiliary_answer': aux_answer
        })
        
        return results
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'main_correct': False,
            'auxiliary_correct': False,
            'total_correct': 0,
            'success_rate': 0,
            'main_answer': None,
            'auxiliary_answer': None
        }]

async def main():
    """Main function for benchmarking mathematical problem solving with both main and auxiliary models."""
    config = BenchmarkConfig.from_args('Benchmark main and auxiliary models on mathematical problems')
    
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
