import os
import asyncio
import logging
import re
from typing import Optional, Dict, List, Tuple
from collections import Counter
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
    """Process a single example using the Engineer-Programmer pipeline approach (one-to-one):
    Multiple engineer prompts, each generating one programming solution"""
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

        # Get models for engineer and programming agents
        main_model = get_model(config, role="main")
        
        # Initialize agents
        engineer_agent = ArchitectAgent(main_model)
        programming_agent = ProgrammingAgent(main_model)
        
        # Generate MULTIPLE engineering analyses and prompts, each with ONE programming solution
        engineer_analyses = []
        programming_solutions = []
        programming_correctness = []
        programming_answers = []
        
        for attempt in range(config.best_of):
            try:
                # Generate a new engineer analysis for each attempt
                logger.append(f"Generating engineering analysis {attempt+1} for problem {running_id + 1}...")
                engineer_prompt, engineer_analysis = await engineer_agent.generate(example["problem"], return_prompt=True)
                engineer_analyses.append(engineer_analysis)
                
                # Extract the response section from the engineer's analysis
                engineer_response = None
                response_match = re.search(r'<response>(.*?)</response>', engineer_analysis, re.DOTALL)
                if response_match:
                    engineer_response = response_match.group(1).strip()
                else:
                    logger.append(f"❌ No response section found in engineer's analysis {attempt+1}")
                    engineer_response = "Please solve this problem using appropriate Python libraries and techniques."
                
                # Combine the original problem with the engineer's guidance
                combined_prompt = PROGRAMMER_SYSTEM_PROMPT+f"Problem:\n{example['problem']}\n\nEngineering Guidance:\n{engineer_response}"
                
                # Generate ONE programming solution for this engineer prompt
                prompt, current_solution = await programming_agent.generate(combined_prompt, return_prompt=True)
                
                # Extract code from solution
                response_match = re.search(r'<response>(.*?)</response>', current_solution, re.DOTALL)
                if response_match:
                    response_content = response_match.group(1)
                    code = extract_code_from_response(response_content)
                    if not code:
                        # If no code in response section, try the whole solution
                        logger.append(f"No code found in response section, trying whole solution")
                        code = extract_code_from_response(current_solution)
                else:
                    # If no response tags, extract from the whole solution
                    code = extract_code_from_response(current_solution)
                
                logger.append(f"Extracted code length: {len(code) if code else 0} characters")
                
                if not code:
                    logger.append(f"❌ No code found in programming solution {attempt+1}")
                    programming_solutions.append(current_solution)
                    programming_correctness.append(False)
                    programming_answers.append(None)
                    continue
                
                # Check code quality
                code_quality_passed, quality_message = check_code_quality(code)
                
                if not code_quality_passed:
                    logger.append(f"❌ Code quality check failed for attempt {attempt+1}: {quality_message}")
                    programming_solutions.append(current_solution)
                    programming_correctness.append(False)
                    programming_answers.append(None)
                    continue
                
                # Run code safely
                execution_success, result, error_message = run_code_safely(code, timeout=config.timeout)
                
                if not execution_success:
                    logger.append(f"❌ Code execution failed for attempt {attempt+1}: {error_message}")
                    programming_solutions.append(current_solution)
                    programming_correctness.append(False)
                    programming_answers.append(None)
                    continue
                
                # Compare with correct answer
                is_correct = False
                try:
                    # Convert correct_answer to float if possible for comparison
                    numeric_correct_answer = None
                    if isinstance(correct_answer, (int, float)):
                        numeric_correct_answer = correct_answer
                    else:
                        try:
                            numeric_correct_answer, _ = extract_numeric_answer(correct_answer)
                        except:
                            pass
                    
                    if numeric_correct_answer is not None and isinstance(result, (int, float)):
                        # Use tolerance for numeric comparison
                        is_correct = abs(numeric_correct_answer - result) <= config.tolerance
                    else:
                        # Try string comparison as fallback
                        is_correct = str(correct_answer).strip() == str(result).strip()
                except Exception as e:
                    logger.append(f"Error comparing answers: {str(e)}")
                
                programming_solutions.append(current_solution)
                programming_correctness.append(is_correct)
                programming_answers.append(result)
                
            except Exception as e:
                logger.append(f"❌ Error in attempt {attempt+1}: {str(e)}")
                engineer_analyses.append(f"Error: {str(e)}")
                programming_solutions.append(f"Error: {str(e)}")
                programming_correctness.append(False)
                programming_answers.append(None)
        
        # Calculate statistics for programming solutions
        programming_success_rate = sum(programming_correctness) / len(programming_correctness) * 100 if programming_correctness else 0
        programming_answer_counts = Counter([str(ans) for ans in programming_answers if ans is not None])
        programming_most_common = programming_answer_counts.most_common(1)
        programming_majority_answer = programming_most_common[0][0] if programming_most_common else None
        programming_majority_correct = any(
            str(ans) == programming_majority_answer and is_correct 
            for ans, is_correct in zip(programming_answers, programming_correctness)
            if ans is not None
        ) if programming_majority_answer else False
        
        # Add statistics to logs
        logger.append("\n" + "="*80)
        logger.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logger.append("="*80)
        logger.append(f"\n📋 Problem:")
        logger.append(f"{example['problem'][:200]}...")
        logger.append(f"\n✓ Expected Answer: {correct_answer}")
        
        # Engineer-Programmer pairs statistics
        logger.append(f"\n📊 Engineer-Programmer Pairs Statistics (One-to-One):")
        for i, (sol_correct, sol_answer) in enumerate(zip(programming_correctness, programming_answers)):
            logger.append(f"├─ Pair {i+1}: {'✓' if sol_correct else '✗'} (Answer: {sol_answer})")
            
            # Show a preview of the engineer's thinking for this pair
            if i < len(engineer_analyses):
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', engineer_analyses[i], re.DOTALL)
                if thinking_match:
                    thinking_content = thinking_match.group(1).strip()
                    # Show just the first line of thinking
                    thinking_preview = thinking_content.split('\n')[0]
                    logger.append(f"│  ├─ Engineer thinking: {thinking_preview}...")
            
            # Add error messages for debugging
            if not sol_correct and i < len(programming_solutions):
                code = extract_code_from_response(programming_solutions[i])
                if not code:
                    logger.append(f"│  └─ No code extracted")
                elif "Error:" in programming_solutions[i]:
                    logger.append(f"│  └─ {programming_solutions[i]}")
        
        logger.append(f"├─ Programming success rate: {programming_success_rate:.1f}%")
        logger.append(f"├─ Programming majority answer: {programming_majority_answer}")
        logger.append(f"└─ Programming majority correct? {'✓' if programming_majority_correct else '✗'}")
        
        logger.append("="*80)
        
        # Print all logs at the end
        logger.print()
        
        # Create result entries
        result_entries = []
        
        # Add engineer and programming training entries (one-to-one pairs)
        for i, (engineer_analysis, solution, is_correct, answer) in enumerate(
            zip(engineer_analyses, programming_solutions, programming_correctness, programming_answers)
        ):
            # Extract engineer response for this pair
            engineer_response = None
            response_match = re.search(r'<response>(.*?)</response>', engineer_analysis, re.DOTALL)
            if response_match:
                engineer_response = response_match.group(1).strip()
            else:
                engineer_response = "Please solve this problem using appropriate Python libraries and techniques."
            
            # Add engineer training entry
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'role': 'engineer',
                'problem': example['problem'],
                'correct_solution': example.get('solution', ''),
                'correct_answer': correct_answer,
                'model_solution': engineer_analysis,
                'is_correct': is_correct,  # Engineer is considered correct if its paired programming solution is correct
                'pair_id': i + 1
            })
            
            # Add programming training entry
            result_entries.append({
                'id': example_id,
                'data_type': 'training',
                'role': 'programmer',
                'problem': example['problem'],
                'engineer_prompt': engineer_response,  # Include the prompt from the engineer
                'correct_solution': example.get('solution', ''),
                'correct_answer': correct_answer,
                'model_solution': solution,
                'model_answer': answer,
                'is_correct': is_correct,
                'pair_id': i + 1
            })
        
        # Add statistics entry
        result_entries.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            # Engineer statistics
            'engineer_count': len(engineer_analyses),  # Multiple engineers in this version
            'has_engineer_thinking': [bool(re.search(r'<thinking>(.*?)</thinking>', ea, re.DOTALL)) for ea in engineer_analyses],
            'has_engineer_response': [bool(re.search(r'<response>(.*?)</response>', ea, re.DOTALL)) for ea in engineer_analyses],
            # Programming solutions statistics
            'programming_solutions_count': len(programming_solutions),
            'programming_correctness': programming_correctness,
            'programming_answers': programming_answers,
            'programming_success_rate': programming_success_rate,
            'programming_majority_answer': programming_majority_answer,
            'programming_majority_correct': programming_majority_correct,
            
            # Compatibility fields for ProgressTracker statistics
            'is_correct_list': programming_correctness,
            'is_most_common_correct': programming_majority_correct,
            'total_solutions': len(programming_solutions),
            'correct_solutions': sum(programming_correctness),
            'incorrect_solutions': len(programming_correctness) - sum(programming_correctness)
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
            'programming_success_rate': None,
            'programming_majority_correct': None
        }]


async def main():
    """Main function for benchmarking with the Engineer-Programmer pipeline approach (one-to-one)."""
    config = BenchmarkConfig.from_args('Benchmark Engineer-Programmer pipeline approach (many engineers → one program each)')
    
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
