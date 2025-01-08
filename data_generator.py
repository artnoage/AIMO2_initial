import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from utils.benchmark_utils import *
from utils.agents import *

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

def generate_analysis_conversation(problem: str, analysis: str) -> Dict:
    """Generate conversation for problem analysis"""
    return {
        'conversations': [
            {
                'content': (
                    "Here is a mathematical problem:\n\n"
                    f"{problem}\n\n"
                    "Please analyze this problem - tell me about its type, "
                    "what theorems and techniques would be useful, and how you'd "
                    "approach solving it. Don't provide the actual solution yet.\n\n"
                    "Start with '**Problem Analysis and Approach**:'"
                ),
                'role': 'user'
            },
            {
                'content': analysis,
                'role': 'assistant'
            }
        ]
    }

def generate_next_step_conversation(problem: str, current_solution: str, next_step: str) -> Dict:
    """Generate conversation for next step prediction"""
    return {
        'conversations': [
            {
                'content': (
                    "Here is a mathematical problem:\n\n"
                    f"{problem}\n\n"
                    "Here's how far we've gotten with the solution:\n\n"
                    f"{current_solution}\n\n"
                    "Could you help me with the next step? Please explain it using LaTeX notation."
                ),
                'role': 'user'
            },
            {
                'content': next_step,
                'role': 'assistant'
            }
        ]
    }

def generate_completion_conversation(problem: str, partial_solution: str, completion: str) -> Dict:
    """Generate conversation for solution completion"""
    return {
        'conversations': [
            {
                'content': (
                    "Here is a mathematical problem:\n\n"
                    f"{problem}\n\n"
                    "We've started solving it and got this far:\n\n"
                    f"{partial_solution}\n\n"
                    "Could you help finish this solution? Remember to put the final answer in \\boxed{}"
                ),
                'role': 'user'
            },
            {
                'content': completion,
                'role': 'assistant'
            }
        ]
    }

def generate_full_solution_conversation(problem: str, solution: str) -> Dict:
    """Generate conversation for complete solution"""
    return {
        'conversations': [
            {
                'content': (
                    "Here is a mathematical problem:\n\n"
                    f"{problem}\n\n"
                    "Could you help me solve this from start to finish? First, let's analyze the problem, "
                    "then walk through the solution step-by-step using LaTeX notation. "
                    "Don't forget to put the final answer in a box using \\boxed{}"
                ),
                'role': 'user'
            },
            {
                'content': solution,
                'role': 'assistant'
            }
        ]
    }

def generate_missing_step_conversation(problem: str, incomplete_solution: str, missing_step: str) -> Dict:
    """Generate conversation for identifying and completing missing steps"""
    return {
        'conversations': [
            {
                'content': (
                    "Here is a mathematical problem:\n\n"
                    f"{problem}\n\n"
                    "I have a solution, but it's missing some steps in between:\n\n"
                    f"{incomplete_solution}\n\n"
                    "Could you help fill in just the missing step? Use LaTeX notation to explain it clearly."
                ),
                'role': 'user'
            },
            {
                'content': missing_step,
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
                
            # Check for unwanted tokens first
            if "[/INST]" in current_solution:
                logs.append(f"✗ Solution contains [/INST] token on attempt {attempts}")
                continue
                
            # Validate solution structure
            is_valid, validation_reason = validate_solution(current_solution)
            if not is_valid:
                logs.append(f"✗ Invalid solution on attempt {attempts}: {validation_reason}")
                continue

            # Quietly check for [/INST] token after validation passes
            if "[/INST]" in current_solution:
                logs.append(f"⚠️ Note: Solution contains [/INST] token but passed validation")

            logs.append(f"✓ Attempt {attempts} passed validation")

            # Verify correctness for valid solutions
            is_correct, reason = await verifier.verify(
                current_solution,
                extract_answer_from_solution(example['solution']),
                example["problem"]
            )
            
            if is_correct:
                final_solution = current_solution
                logs.append(f"✓ Found valid and correct solution on attempt {attempts}")
                break
                
        except Exception as e:
            print(f"Error in solution attempt {attempts}: {str(e)}")
            continue

    if not final_solution:
        return None


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
        solver = get_model(config, role="solver")
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

        prompt, solution, solution_logs = result
        logs.append(solution_logs)  # Add solution logs to main logs

        # Add final summary to logs
        logs.append("\n" + "="*50)
        logs.append("📊 Final Summary:")
        processing_time = time.perf_counter() - start_time
        logs.append(f"├─ Processing time: {processing_time:.2f}s")
        logs.append("="*50)

        # Generate all training variants from the valid solution
        results = []
        steps = split_into_steps(solution)
        
        # Return empty list if solution has less than 3 steps (including analysis)
        if len(steps) < 4:
            logs.append("\n⚠️ Solution has fewer than 4 steps - skipping")
            print("\n".join(logs))
            return []
            
        # 1. Analysis conversation - use first part if it contains analysis
        if steps and "analysis" in steps[0].lower():
            results.append({
                'id': f"{example_id}_analysis",
                'problem': example['problem'],
                'correct_answer': correct_answer,
                **generate_analysis_conversation(example['problem'], steps[0])
            })
        
        # 2. Full solution conversation
        results.append({
            'id': f"{example_id}_full",
            'problem': example['problem'],
            'correct_answer': correct_answer,
            **generate_full_solution_conversation(example['problem'], solution)
        })
        
        # 3. Step-by-step training data
        if len(steps) > 1:
            for i in range(len(steps)-1):
                current = "\n\n".join(steps[:i+1])
                next_step = steps[i+1]  # Use the next step from our valid solution
                results.append({
                    'id': f"{example_id}_step_{i+1}",
                    'problem': example['problem'],
                    'correct_answer': correct_answer,
                    **generate_next_step_conversation(example['problem'], current, next_step)
                })
        
        # 4. Completion training data
        partial_solutions = get_partial_solutions(steps)
        if len(partial_solutions) > 1:
            for i in range(len(partial_solutions)-1):
                partial = partial_solutions[i]
                completion = "\n\n".join(steps[i+1:])  # Use remaining steps from our valid solution
                results.append({
                    'id': f"{example_id}_complete_{i+1}",
                    'problem': example['problem'],
                    'correct_answer': correct_answer,
                    **generate_completion_conversation(example['problem'], partial, completion)
                })
                
        # 5. Missing step training data
        if len(steps) > 2:  # Need at least 3 steps to have meaningful missing steps
            for i in range(1, len(steps)-1):  # Skip first and last steps
                # Insert [missing_step] token where the step was removed
                incomplete = "\n\n".join(steps[:i] + ["[missing_step]"] + steps[i+1:])
                missing_step = steps[i]  # Use the removed step from our valid solution
                results.append({
                    'id': f"{example_id}_missing_{i}",
                    'problem': example['problem'],
                    'correct_answer': correct_answer,
                    **generate_missing_step_conversation(example['problem'], incomplete, missing_step)
                })
        
        # Print all logs at the end
        print("\n".join(logs))
        
        return results

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
