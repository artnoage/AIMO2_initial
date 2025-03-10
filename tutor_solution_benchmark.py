import os
import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.model_utils import *
from utils.solution_utils import *
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
    """Process a single example using both solution and tutor agents"""
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

        # Get models for solution and tutor agents
        main_model = get_model(config, role="main")
        tutor_model = get_model(config, role="main") 
        
        # Initialize agents
        solution_agent = FullSolutionAgent(main_model)
        tutor_agent = TutorAgent(main_model)
        
        # Generate initial solution
        prompt, initial_solution = await solution_agent.generate(example["problem"], return_prompt=True)
        
        # Have tutor evaluate the solution
        tutor_response = await tutor_agent.find_first_wrong_step(
            example["problem"],
            initial_solution,
            return_prompt=False
        )
        
        # Extract verdict from tutor response
        verdict = "The answer is correct"  # Default assumption
        corrected_solution = None
        
        # Parse tutor response to find verdict and corrected solution
        if "<verdict>" in tutor_response and "</verdict>" in tutor_response:
            verdict_start = tutor_response.find("<verdict>") + len("<verdict>")
            verdict_end = tutor_response.find("</verdict>")
            verdict = tutor_response[verdict_start:verdict_end].strip()
        
        # Extract corrected solution if available
        if "<finalization>" in tutor_response and "</finalization>" in tutor_response:
            finalization_start = tutor_response.find("<finalization>") + len("<finalization>")
            finalization_end = tutor_response.find("</finalization>")
            corrected_solution = tutor_response[finalization_start:finalization_end].strip()
            
            # If finalization is empty, set to None
            if not corrected_solution or corrected_solution.isspace():
                corrected_solution = None
        
        # Determine which solution to use
        final_solution = initial_solution
        solution_source = "original"
        
        if "The answer is correct" not in verdict and corrected_solution:
            # Extract answer from corrected solution
            corrected_answer = extract_answer_from_solution(corrected_solution)
            
            if corrected_answer is not None:
                final_solution = corrected_solution
                solution_source = "corrected"
        
        # Verify the final solution
        verifier = NumericVerifier(tolerance=config.tolerance)
        is_correct, final_answer = await verifier.verify(
            final_solution,
            correct_answer,
            example["problem"]
        )
        
        # Get initial solution correctness
        initial_is_correct, initial_answer = await verifier.verify(initial_solution, correct_answer, example['problem'])
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Statistics:")
        logger.append(f"├─ Initial solution correct? {initial_is_correct}")
        logger.append(f"├─ Initial answer: {initial_answer}")
        logger.append(f"├─ Tutor verdict: {verdict}")
        logger.append(f"├─ Solution used: {solution_source}")
        logger.append(f"├─ Final answer: {final_answer}")
        logger.append(f"└─ Final solution correct? {is_correct}")
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add detailed entry
        result_entries.append({
            'id': example_id,
            'data_type': 'training',
            'problem': example['problem'],
            'correct_solution': example['solution'],
            'correct_answer': correct_answer,
            'initial_solution': initial_solution,
            'tutor_response': tutor_response,
            'tutor_verdict': verdict,
            'corrected_solution': corrected_solution,
            'final_solution': final_solution,
            'final_answer': final_answer,
            'solution_source': solution_source,
            'is_correct': is_correct
        })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'initial_solution_correct': initial_is_correct,
            'tutor_verdict_correct': ("The answer is correct" in verdict) == initial_is_correct,
            'final_solution_correct': is_correct,
            'solution_source': solution_source,
            'solution_improved': not initial_is_correct and is_correct,
            'solution_worsened': initial_is_correct and not is_correct
        })
        
        return result_entries
        
    except Exception as e:
        logger.append(f"❌ Error processing example {str(running_id)}: {e}")
        import traceback
        logger.append(traceback.format_exc())
        logger.print()
        return [{
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': False,
            'initial_solution_correct': None,
            'tutor_verdict_correct': None,
            'final_solution_correct': None,
            'solution_source': None
        }]


async def main():
    """Main function for benchmarking with solution and tutor agents combined."""
    config = BenchmarkConfig.from_args('Benchmark solution and tutor agents combined')
    
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
