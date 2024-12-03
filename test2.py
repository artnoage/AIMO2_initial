import os
import json
import asyncio
import re
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dotenv import load_dotenv
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import run_benchmark
from utils.agents import AnalysisAgent, NextStepAgent, CompletionAgent
from langchain_core.messages import HumanMessage, SystemMessage
from benchmark_numeric import verify_numeric

def count_solution_steps(solution: str) -> int:
    """Count the number of steps in a solution by looking for 'Step X' patterns"""
    # Look for patterns like "Step 1", "Step 2", etc.
    steps = re.findall(r'Step\s+\d+', solution, re.IGNORECASE)
    return len(steps)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, best_of: int, initial_steps: int = 0) -> Optional[Dict]:
    """Process a single example using hybrid approach: analysis + initial_steps + completion"""
    try:
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            print(f"Error processing example {running_id}: Invalid example format")
            return None
            
        correct_answer = extract_answer_from_solution(example['solution'])
        if correct_answer is None:
            print(f"Warning: Could not extract answer from solution for example {running_id}")
            return None

        # Initialize agents
        analysis_agent = AnalysisAgent(solver_model)
        step_agent = NextStepAgent(solver_model)
        completion_agent = CompletionAgent(solver_model)
        
        solutions = []
        correct_count = 0
        
        for attempt in range(best_of):
            try:
                # Start with analysis
                current_solution = await analysis_agent.generate(example["problem"])
                current_solution = current_solution.content
                steps_taken = 0
                complete_solution = current_solution
                for step in range(initial_steps):
                    steps_taken += 1
                    next_step = await step_agent.generate(
                        example["problem"], 
                        HumanMessage(content=current_solution)
                    )
                    step_content = next_step.content if hasattr(next_step, 'content') else str(next_step)
                    current_solution = f"{current_solution}\n\n{step_content}"
                    # Check if we already have an answer
                    if extract_answer_from_solution(current_solution) is not None:
                        print("answer found in step", step + 1, "for attempt", attempt, "in problem", running_id)
                        complete_solution = current_solution
                        break
                else:
                    # Complete the solution if we didn't find an answer in the first two steps
                    steps_taken += 1
                    completion = await completion_agent.generate(example["problem"], current_solution)
                    completion_content = completion.content if hasattr(completion, 'content') else str(completion)
                    complete_solution = completion_content

                # Extract and verify answer
                current_answer = extract_answer_from_solution(complete_solution)
                
                if current_answer is not None:
                    # Verify the numeric answer
                    current_answer_float, is_correct = await verify_numeric(complete_solution, correct_answer, 1e-6)
                    if is_correct:
                        correct_count += 1
                else:
                    is_correct = False
                
                solutions.append({
                    'solution': current_solution,
                    'complete_solution': complete_solution,
                    'answer': current_answer,
                    'is_correct': is_correct,
                    'steps_before_completion': steps_taken,
                    'total_steps': count_solution_steps(complete_solution)
                })
                    
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                solutions.append({
                    'solution': "Error occurred",
                    'answer': None,
                    'is_correct': False,
                    'steps': 0
                })
        
        # Print statistics
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Steps before completion: {[s['steps_before_completion'] for s in solutions]}")
        print(f"Total solution steps: {[s['total_steps'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{best_of}")
        print(f"Success rate: {(correct_count/best_of)*100:.1f}%")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['complete_solution'].split("Problem:")[0].strip() for s in solutions],  # Remove metadata
            'intermediate_solutions': [s['solution'].split("Problem:")[0].strip() for s in solutions],  # Intermediate steps
            'model_answers': [s['answer'] for s in solutions],
            'steps_before_completion': [s['steps_before_completion'] for s in solutions],
            'total_solution_steps': [s['total_steps'] for s in solutions],
            'is_correct_list': [s['is_correct'] for s in solutions],
            'correct_binary': [1 if s['is_correct'] else 0 for s in solutions],
            'model_answer_raw': solutions[0]['answer'],
            'correct_answer_raw': correct_answer,
            'attempts': {
                'total': len(solutions),
                'correct_count': correct_count
            }
        }
        
    except Exception as e:
        print(f"Error processing example {running_id}: {e}")
        return None

async def main():
    """Main function for testing hybrid solution generation."""
    config = NumericConfig.from_args('Test hybrid solution generation')
    results = await run_benchmark(config, 
                                lambda ex, rid, eid, sm, vm, bo: process_example(ex, rid, eid, sm, vm, bo))
    return results

if __name__ == "__main__":
    try:
        all_results = asyncio.run(main())
        # Filter out None results and prepare final output
        final_results = [r for r in all_results if r is not None]
        
        # Save results to JSON file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test2_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
