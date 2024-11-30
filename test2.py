import os
import asyncio
from typing import Optional, Dict, List
from dotenv import load_dotenv
from utils.utils import *
from utils.benchmark_config import *
from utils.benchmark_utils import run_benchmark
from utils.agents import AnalysisAgent, NextStepAgent, CompletionAgent
from langchain_core.messages import HumanMessage, SystemMessage

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
                
                # Generate first two steps individually
                for step in range(2):
                    next_step = await step_agent.generate(example["problem"], current_solution)
                    current_solution = f"{current_solution}\n\n{next_step}"
                
                # Complete the solution
                complete_solution = await completion_agent.generate(example["problem"], current_solution)
                
                # Extract and verify answer
                current_answer = extract_answer_from_solution(complete_solution)
                
                if current_answer is not None:
                    is_correct = await compare_math_answers(
                        current_answer,
                        correct_answer,
                        example["problem"],
                        verifier_model
                    )
                    
                    if is_correct:
                        correct_count += 1
                else:
                    is_correct = False
                
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
    config = BenchmarkConfig.from_args('Test hybrid solution generation')
    verifier_model = get_model(ModelOption[config.verifier], temp=config.verifier_temp)
    await run_benchmark(config, 
                       process_example,
                       BENCHMARK_SYSTEM_PROMPT,
                       verifier_model)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
