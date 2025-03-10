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
        auxiliary_model = get_model(config, role="auxiliary")
        
        # Initialize agents
        solution_agent = FullSolutionAgent(main_model)
        tutor_agent = TutorAgent(auxiliary_model)
        
        # Generate multiple initial solutions if best_of > 1
        initial_solutions = []
        initial_correctness = []
        initial_answers = []
        
        for attempt in range(config.best_of):
            prompt, current_solution = await solution_agent.generate(example["problem"], return_prompt=True)
            
            # Verify the solution
            verifier = NumericVerifier(tolerance=config.tolerance)
            is_correct, answer = await verifier.verify(
                current_solution,
                correct_answer,
                example["problem"]
            )
            
            initial_solutions.append(current_solution)
            initial_correctness.append(is_correct)
            initial_answers.append(answer)
        
        # Have tutor evaluate all solutions
        tutor_responses = []
        tutor_verdicts = []
        corrected_solutions = []
        final_solutions = []
        final_correctness = []
        final_answers = []
        solution_sources = []
        
        for i, initial_solution in enumerate(initial_solutions):
            # Have tutor evaluate the solution
            tutor_response = await tutor_agent.find_first_wrong_step(
                example["problem"],
                initial_solution,
                return_prompt=False
            )
            tutor_responses.append(tutor_response)
            
            # Extract verdict from tutor response
            verdict = "The answer is correct"  # Default assumption
            corrected_solution = None
            
            # Parse tutor response to find verdict and corrected solution
            if "<verdict>" in tutor_response and "</verdict>" in tutor_response:
                verdict_start = tutor_response.find("<verdict>") + len("<verdict>")
                verdict_end = tutor_response.find("</verdict>")
                verdict = tutor_response[verdict_start:verdict_end].strip()
            
            tutor_verdicts.append(verdict)
            
            # Extract corrected solution if available
            if "<finalization>" in tutor_response and "</finalization>" in tutor_response:
                finalization_start = tutor_response.find("<finalization>") + len("<finalization>")
                finalization_end = tutor_response.find("</finalization>")
                corrected_solution_text = tutor_response[finalization_start:finalization_end].strip()
                
                # If finalization is empty, set to None
                if not corrected_solution_text or corrected_solution_text.isspace():
                    corrected_solution_text = None
                    
                corrected_solutions.append(corrected_solution_text)
            else:
                corrected_solutions.append(None)
            
            # Determine which solution to use
            final_solution = initial_solution
            solution_source = "original"
            
            if "The answer is correct" not in verdict and corrected_solution_text:
                # Extract answer from corrected solution
                corrected_answer = extract_answer_from_solution(corrected_solution_text)
                
                if corrected_answer is not None:
                    final_solution = corrected_solution_text
                    solution_source = "corrected"
            
            final_solutions.append(final_solution)
            solution_sources.append(solution_source)
            
            # Verify the final solution
            is_correct, final_answer = await verifier.verify(
                final_solution,
                correct_answer,
                example["problem"]
            )
            
            final_correctness.append(is_correct)
            final_answers.append(final_answer)
        
        # Calculate majority votes
        from collections import Counter
        
        # Initial solutions majority vote
        initial_majority_correct = sum(initial_correctness) > len(initial_correctness) / 2
        
        # Get most common initial answer
        if any(isinstance(a, (int, float)) for a in initial_answers if a is not None):
            # For numeric answers, use the median
            numeric_answers = [a for a in initial_answers if a is not None and isinstance(a, (int, float))]
            initial_majority_answer = sorted(numeric_answers)[len(numeric_answers)//2] if numeric_answers else None
        else:
            # For non-numeric answers, use the most common
            initial_majority_answer = Counter([str(a) for a in initial_answers if a is not None]).most_common(1)[0][0] if initial_answers else None
        
        # Final solutions majority vote
        final_majority_correct = sum(final_correctness) > len(final_correctness) / 2
        
        # Get most common final answer
        if any(isinstance(a, (int, float)) for a in final_answers if a is not None):
            # For numeric answers, use the median
            numeric_answers = [a for a in final_answers if a is not None and isinstance(a, (int, float))]
            final_majority_answer = sorted(numeric_answers)[len(numeric_answers)//2] if numeric_answers else None
        else:
            # For non-numeric answers, use the most common
            final_majority_answer = Counter([str(a) for a in final_answers if a is not None]).most_common(1)[0][0] if final_answers else None
        
        # Verify majority answers
        initial_majority_is_correct = False
        final_majority_is_correct = False
        
        if initial_majority_answer is not None:
            initial_majority_is_correct, _ = await verifier.verify(
                f"\\boxed{{{initial_majority_answer}}}", 
                correct_answer,
                example["problem"]
            )
            
        if final_majority_answer is not None:
            final_majority_is_correct, _ = await verifier.verify(
                f"\\boxed{{{final_majority_answer}}}", 
                correct_answer,
                example["problem"]
            )
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        logger.append(f"\n📊 Initial Solutions Statistics:")
        
        # Log all initial solutions
        for i, (sol_correct, sol_answer) in enumerate(zip(initial_correctness, initial_answers)):
            logger.append(f"├─ Solution {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
        
        # Calculate initial success rate
        initial_success_rate = sum(initial_correctness) / len(initial_correctness) * 100 if initial_correctness else 0
        logger.append(f"├─ Initial success rate: {initial_success_rate:.1f}%")
        logger.append(f"├─ Initial majority vote correct? {'✓' if initial_majority_is_correct else '✗'}")
        logger.append(f"├─ Initial majority answer: {initial_majority_answer}")
        
        logger.append(f"\n📊 Tutor Evaluations:")
        
        # Log all tutor evaluations
        for i, (verdict, source, final_correct, final_ans) in enumerate(zip(tutor_verdicts, solution_sources, final_correctness, final_answers)):
            logger.append(f"├─ Solution {i+1}: Verdict: {verdict[:30]}... | Source: {source} | Correct: {'✓' if final_correct else '✗'} | Answer: {final_ans}")
        
        # Calculate final success rate
        final_success_rate = sum(final_correctness) / len(final_correctness) * 100 if final_correctness else 0
        logger.append(f"├─ Final success rate: {final_success_rate:.1f}%")
        logger.append(f"├─ Final majority vote correct? {'✓' if final_majority_is_correct else '✗'}")
        logger.append(f"├─ Final majority answer: {final_majority_answer}")
        logger.append(f"└─ Improvement from tutor corrections? {'✓' if final_majority_is_correct and not initial_majority_is_correct else '✗'}")
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
            'initial_solutions': initial_solutions,
            'initial_correctness': initial_correctness,
            'initial_answers': initial_answers,
            'tutor_responses': tutor_responses,
            'tutor_verdicts': tutor_verdicts,
            'corrected_solutions': corrected_solutions,
            'final_solutions': final_solutions,
            'final_correctness': final_correctness,
            'final_answers': final_answers,
            'solution_sources': solution_sources
        })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'initial_solutions_count': len(initial_solutions),
            'initial_correctness': initial_correctness,
            'initial_answers': initial_answers,
            'initial_success_rate': initial_success_rate,
            'initial_majority_correct': initial_majority_is_correct,
            'initial_majority_answer': initial_majority_answer,
            'tutor_verdicts': tutor_verdicts,
            'final_correctness': final_correctness,
            'final_answers': final_answers,
            'final_success_rate': final_success_rate,
            'final_majority_correct': final_majority_is_correct,
            'final_majority_answer': final_majority_answer,
            'solution_sources': solution_sources,
            'majority_vote_improved': final_majority_is_correct and not initial_majority_is_correct,
            'majority_vote_worsened': initial_majority_is_correct and not final_majority_is_correct,
            'success_rate_improved': final_success_rate > initial_success_rate
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
