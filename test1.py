import os
import json
import asyncio
import re
from typing import Optional, Dict, List
from datetime import datetime
from dotenv import load_dotenv
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import run_benchmark
from utils.agents import AnalysisAgent, NextStepAgent
from benchmark_numeric import verify_numeric

def normalize_latex(text: str) -> str:
    """Replace more than two backslashes with two backslashes to fix excessive escaping"""
    return re.sub(r'\\{3,}', r'\\\\', text)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, best_of: int) -> Optional[Dict]:
    """Process a single example using sequential agents"""
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
        
        solutions = []
        correct_count = 0
        max_steps = 20  # Maximum steps to prevent infinite loops
        
        for attempt in range(best_of):
            try:
                # Start with analysis
                current_solution = await analysis_agent.generate(example["problem"])
                steps_taken = 0
                has_answer = False
                current_solution = normalize_latex(current_solution.content)
                # Keep adding steps until we get an answer or hit max steps
                while not has_answer and steps_taken < max_steps:
                    steps_taken += 1
                    # Get next step
                    next_step = await step_agent.generate(
                        example["problem"], 
                        current_solution
                    )
                    
                    # Handle AIMessage or string content
                    step_content = next_step.content if hasattr(next_step, 'content') else str(next_step)
                    step_content = normalize_latex(step_content)
                    current_solution = current_solution + step_content
                    
                    # Check if we have an answer
                    current_answer = extract_answer_from_solution(current_solution)
                    has_answer = current_answer is not None
                    
                    if has_answer:
                        # Verify the numeric answer
                        current_answer_float, is_correct = await verify_numeric(current_solution, correct_answer, 1e-6)
                        
                        if is_correct:
                            correct_count += 1
                            
                        solutions.append({
                            'solution': current_solution,
                            'answer': current_answer,
                            'is_correct': is_correct,
                            'steps': steps_taken
                        })
                        break
                
                if not has_answer:
                    # If we hit max steps without an answer
                    solutions.append({
                        'solution': current_solution,
                        'answer': None,
                        'is_correct': False,
                        'steps': steps_taken
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
        print(f"Steps taken: {[s['steps'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{best_of}")
        print(f"Success rate: {(correct_count/best_of)*100:.1f}%")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['solution'].split("Problem:")[0].strip() for s in solutions],  # Remove metadata
            'model_answers': [s['answer'] for s in solutions],
            'steps_taken': [s['steps'] for s in solutions],
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
    """Main function for testing step-by-step solution generation."""
    config = NumericConfig.from_args('Test step-by-step solution generation')
    await run_benchmark(config, 
                       lambda ex, rid, eid, sm, vm, bo: process_example(ex, rid, eid, sm, None, bo),
                       BENCHMARK_SYSTEM_PROMPT,
                       None)

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        if results:
            # Filter out None values and clean up data for JSON serialization
            cleaned_results = [r for r in results if r is not None]
            for result in cleaned_results:
                # Convert any None values in lists to "null" strings
                if 'model_answers' in result:
                    result['model_answers'] = ["null" if x is None else x for x in result['model_answers']]
                if 'model_responses' in result:
                    result['model_responses'] = ["null" if x is None else x for x in result['model_responses']]

            # Save results to JSON file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test1_results_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(cleaned_results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        else:
            print("\nNo results were generated")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
