import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from bench_utils.benchmark_config import *
from bench_utils.benchmark_utils import *
from bench_utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

def generate_next_step_conversation(problem: str, current_solution: str, next_step: str) -> Dict:
    """Generate conversation for next step prediction"""
    return {
        'conversations': [
            {
                'content': f"Here is a mathematical problem:\n\n{problem}\n\n"
                          f"Current solution:\n\n{current_solution}\n\n"
                          "What is the next step in this solution?",
                'role': 'user'
            },
            {
                'content': next_step,
                'role': 'assistant'
            }
        ]
    }

def generate_complete_from_partial(problem: str, partial_solution: str, complete_solution: str) -> Dict:
    """Generate conversation for completing partial solutions"""
    return {
        'conversations': [
            {
                'content': f"Here is a mathematical problem:\n\n{problem}\n\n"
                          f"Here is a partial solution:\n\n{partial_solution}\n\n"
                          "Please complete this solution.",
                'role': 'user'
            },
            {
                'content': complete_solution,
                'role': 'assistant'
            }
        ]
    }

async def process_solution(
    example: Dict,
    solver: any,
    verifier: any,
    config: BenchmarkConfig
) -> Optional[List[Dict]]:
    """Process example to generate multiple training data variants for SFT"""
    logs = []
    solution_agent = FullSolutionAgent(solver)
    solution_prompt = None
    final_solution = None
    
    attempts = 0
    while attempts < config.best_of:
        attempts += 1
        try:
            if solution_prompt is None:
                solution_prompt, current_solution = await solution_agent.generate(
                    example["problem"], return_prompt=True)
            else:
                current_solution = await solution_agent.generate(example["problem"])
                
            # Validate solution structure
            is_valid, validation_reason = validate_solution(current_solution)
            if not is_valid:
                logs.append(f"✗ Invalid solution on attempt {attempts}: {validation_reason}")
                continue

            logs.append(f"✓ Attempt {attempts} passed validation")

            # Verify correctness for valid solutions
            is_correct, reason = await verifier.verify(
                current_solution,
                extract_answer_from_solution(example['solution']),
                example["problem"]
            )
            
            if is_correct:
                final_solution = current_solution
                logs.append(f"✓ Found valid solution on attempt {attempts}")
                break
                
        except Exception as e:
            print(f"Error in solution attempt {attempts}: {str(e)}")
            continue

    if not final_solution:
        return None

    # Print summary
    print(f"\nExample completed: Found valid solution in {attempts} attempts")

    # Add summary to logs
    logs.append("\n" + "="*50)
    logs.append("=== Solution Details ===")
    logs.append("="*50)
    logs.append(f"\n✓ Found valid solution in {attempts} attempts")
    
    return (
        solution_prompt,
        final_solution,
        "\n".join(logs)
    )

async def process_example(
    example: Dict,
    running_id: int,
    example_id: int,
    config: BenchmarkConfig
) -> Optional[Dict]:
    """Process a single example for SFT data generation"""
    start_time = time.perf_counter()
    logs = []

    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None

        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Initialize models and verifier
        solver = get_model(ModelOption[config.solver], temp=config.temperature)
        verifier = NumericVerifier(tolerance=config.tolerance)

        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)

        # Problem details
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n✓ Expected Answer: {correct_answer}")

        result = await process_solution(example, solver, verifier, config)
        if not result:
            return None

        prompt, solution, quality_score, solution_logs = result
        print(solution_logs)  # Print the logs from solution generation

        # Add final summary to logs
        logs.append("\n" + "="*50)
        logs.append("📊 Final Summary:")
        processing_time = time.perf_counter() - start_time
        logs.append(f"├─ Processing time: {processing_time:.2f}s")
        logs.append(f"├─ Quality score: {quality_score:.3f}")
        logs.append("="*50)

        # Print logs before returning result
        print("\n".join(logs))

        # Return consistent format for SFT data with conversation tags
        result = {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'conversations': [
                {
                    'content': f"Here is a mathematical problem to solve:\n\n{example['problem']}\n\n"
                              "Please provide a complete solution with analysis and steps.",
                    'role': 'user'
                },
                {
                    'content': solution,
                    'role': 'assistant'
                }
            ]
        }
        return [result]

    except Exception as e:
        processing_time = time.perf_counter() - start_time
        error_category = (
            "timeout" if isinstance(e, TimeoutError)
            else "validation" if isinstance(e, ValueError)
            else "rate_limit" if "rate limit" in str(e).lower()
            else "context_length" if "context length" in str(e).lower()
            else "other"
        )
        error_details = {
            'id': example_id,
            'status': 'error',
            'error_type': type(e).__name__,
            'error_message': str(e),
            'error_category': error_category,
            'processing_time': processing_time,
            'logs': "\n".join(logs)
        }
        logging.error(f"\n❌ Error processing example {running_id}:")
        logging.error(f"├─ Error type: {error_details['error_type']}")
        logging.error(f"├─ Error message: {error_details['error_message']}")
        logging.error(f"├─ Processing time: {processing_time:.2f}s")
        logging.error(f"└─ Example ID: {example_id}")
        return None

async def main():
    """Main function for SFT data generation."""
    config = BenchmarkConfig.from_args('SFT data generation')
    await run_benchmark(
        config=config,
        process_example_func=process_example
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nData generation interrupted by user")
    except Exception as e:
        print(f"\nData generation failed with error: {e}")
