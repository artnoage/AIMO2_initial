import os
import asyncio
from typing import Optional, Dict, List
from dotenv import load_dotenv
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import run_benchmark
from utils.agents import AnalysisAgent, NextStepAgent, CompletionAgent
from langchain_core.messages import HumanMessage, SystemMessage
from benchmark_numeric import verify_numeric

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

async def process_example(example: Dict, running_id: int, example_id: int, solver_model, verifier_model, best_of: int) -> Optional[Dict]:
    """Process a single example using hybrid approach: analysis + 2 steps + completion"""
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
                
                # Generate first two steps individually (unless we get an answer sooner)
                for step in range(2):
                    next_step = await step_agent.generate(example["problem"], current_solution)
                    current_solution = f"{current_solution}\n\n{next_step}"
                    
                    # Check if we already have an answer
                    if extract_answer_from_solution(current_solution) is not None:
                        complete_solution = current_solution
                        break
                else:
                    # Complete the solution if we didn't find an answer in the first two steps
                    complete_solution = await completion_agent.generate(example["problem"], current_solution)
                
                # Extract and verify answer
                current_answer = extract_answer_from_solution(complete_solution)
                
                # Verify the numeric answer
                current_answer_float, is_correct = await verify_numeric(complete_solution, correct_answer, 1e-6)
                if is_correct:
                    correct_count += 1
                
                solutions.append({
                    'solution': complete_solution,
                    'answer': current_answer,
                    'is_correct': is_correct
                })
                    
            except Exception as e:
                print(f"Error in attempt {attempt + 1} for example {running_id}: {str(e)}")
                solutions.append({
                    'solution': "Error occurred",
                    'answer': None,
                    'is_correct': False
                })
        
        # Print statistics
        print(f"\nExample {running_id + 1}:")
        print(f"Problem: {example['problem'][:200]}...")
        print(f"Correct answer: {correct_answer}")
        print(f"Model answers: {[s['answer'] for s in solutions]}")
        print(f"Correct/incorrect: {[1 if s['is_correct'] else 0 for s in solutions]}")
        print(f"Correct solutions: {correct_count}/{best_of}")
        print(f"Success rate: {(correct_count/best_of)*100:.1f}%")
        print("-" * 80)
        
        return {
            'id': example_id,
            'problem': example['problem'],
            'correct_answer': correct_answer,
            'model_responses': [s['solution'] for s in solutions],
            'model_answers': [s['answer'] for s in solutions],
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
    await run_benchmark(config, 
                       lambda ex, rid, eid, sm, vm, bo: process_example(ex, rid, eid, sm, None, bo),
                       BENCHMARK_SYSTEM_PROMPT,
                       None)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
