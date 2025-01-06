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

def generate_analysis_conversation(problem: str, analysis: str) -> Dict:
    """Generate conversation for problem analysis"""
    return {
        'conversations': [
            {
                'content': (
                    f"Here is a mathematical problem:\n\n{problem}\n\n"
                    "Before solving this problem step-by-step, provide a thorough analysis that:\n"
                    "1. Categorizes the problem type\n"
                    "2. Lists the specific theorems and techniques that will be useful\n"
                    "3. Outlines the general approach to solving it\n\n"
                    "Important guidelines:\n"
                    "- Start with '**Problem Analysis and Approach**:'\n"
                    "- Be specific about which theorems/techniques apply\n"
                    "- Explain why these approaches are suitable\n"
                    "- Do NOT provide the actual solution steps"
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
                    f"Here is a mathematical problem:\n\n{problem}\n\n"
                    "Your task is to provide the next step in the solution. "
                    "Make sure your step is detailed and mathematically rigorous.\n\n"
                    "Guidelines:\n"
                    "- Provide exactly ONE step\n"
                    "- Include clear explanations\n"
                    "- Use LaTeX notation where appropriate\n"
                    "- Include justification in [brackets]\n"
                    "- Number your step appropriately\n\n"
                    f"Here are the steps so far:\n\n{current_solution}\n\n"
                    "Provide the next step:"
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
                    f"Here is a mathematical problem:\n\n{problem}\n\n"
                    "I will show you the beginning of a step-by-step mathematical solution. "
                    "Your task is to complete the solution by continuing with the same style and rigor.\n\n"
                    "Important guidelines:\n"
                    "- Maintain the same level of detail and explanation as the previous steps\n"
                    "- Continue the step numbering sequence\n"
                    "- Use LaTeX notation consistently\n"
                    "- Provide justification for each step in [brackets]\n"
                    "- End with a clear boxed answer using \\boxed{}\n\n"
                    f"Here is the partial solution:\n\n{partial_solution}\n\n"
                    "Please complete the remaining steps following the same format:"
                ),
                'role': 'user'
            },
            {
                'content': completion,
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
                    f"Problem:\n\n{problem}\n\n"
                    "Here is a solution that may be missing important intermediate steps:\n\n"
                    f"{incomplete_solution}\n\n"
                    "Your task:\n"
                    "1. Identify where additional explanation or steps are needed\n"
                    "2. Generate ONLY the missing step(s) that would make the solution clearer\n"
                    "3. Match the style and notation of the existing solution\n"
                    "4. Include full mathematical notation and justification in [brackets]\n"
                    "5. Make sure the step fits logically between the surrounding steps\n\n"
                    "Generate ONLY the missing step(s):"
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

        # Generate all training variants
        results = []
        
        # 1. Analysis conversation
        analysis_agent = AnalysisAgent(solver)
        analysis = await analysis_agent.generate(example['problem'])
        results.append({
            'id': f"{example_id}_analysis",
            'problem': example['problem'],
            'correct_answer': correct_answer,
            **generate_analysis_conversation(example['problem'], analysis)
        })
        
        # 2. Full solution conversation
        results.append({
            'id': f"{example_id}_full",
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'conversations': [
                {
                    'content': (
                        f"Here is a mathematical problem to solve:\n\n{example['problem']}\n\n"
                        "Please provide a complete solution following these guidelines:\n"
                        "1. Start with '**Problem Analysis and Approach**:' section explaining:\n"
                        "   - Problem type and key concepts involved\n"
                        "   - Relevant theorems and techniques\n"
                        "   - Overall solution strategy\n\n"
                        "2. Then provide a detailed step-by-step solution:\n"
                        "   - Number each step clearly (Step 1, Step 2, etc.)\n"
                        "   - Show all work and intermediate calculations\n"
                        "   - Use LaTeX notation for mathematical expressions\n"
                        "   - Provide justification in [brackets] for key steps\n"
                        "   - End with final answer in \\boxed{}"
                    ),
                    'role': 'user'
                },
                {
                    'content': solution,
                    'role': 'assistant'
                }
            ]
        })
        
        # 3. Generate step-by-step training data
        next_step_agent = NextStepAgent(solver)
        steps = split_into_steps(solution)
        if len(steps) > 1:
            for i in range(len(steps)-1):
                current = "\n\n".join(steps[:i+1])
                next_step = await next_step_agent.generate(example['problem'], current)
                results.append({
                    'id': f"{example_id}_step_{i+1}",
                    'problem': example['problem'],
                    'correct_answer': correct_answer,
                    **generate_next_step_conversation(example['problem'], current, next_step)
                })
        
        # 4. Generate completion training data
        completion_agent = CompletionAgent(solver)
        partial_solutions = get_partial_solutions(steps)
        if len(partial_solutions) > 1:
            for i in range(len(partial_solutions)-1):
                partial = partial_solutions[i]
                completion = await completion_agent.generate(example['problem'], partial)
                results.append({
                    'id': f"{example_id}_complete_{i+1}",
                    'problem': example['problem'],
                    'correct_answer': correct_answer,
                    **generate_completion_conversation(example['problem'], partial, completion)
                })
                
        # 5. Generate missing step training data
        missing_step_agent = MissingStepAgent(solver)
        if len(steps) > 2:  # Need at least 3 steps to have meaningful missing steps
            for i in range(1, len(steps)-1):  # Skip first and last steps
                incomplete = "\n\n".join(steps[:i] + steps[i+1:])  # Remove one step
                missing_step = await missing_step_agent.complete_missing_step(example['problem'], incomplete)
                results.append({
                    'id': f"{example_id}_missing_{i}",
                    'problem': example['problem'],
                    'correct_answer': correct_answer,
                    **generate_missing_step_conversation(example['problem'], incomplete, missing_step)
                })
        
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
