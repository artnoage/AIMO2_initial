import os
import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import get_model, extract_answer_from_solution, split_into_steps
from utils.agents import TutorAgent
from utils.logger import BenchmarkLogger
from collections import Counter
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

def extract_sections(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract the Analysis, Verdict and Substitution sections from the response"""
    analysis_match = re.search(r'</Analysis>\s*(.*?)\s*<Analysis>', response, re.DOTALL)
    verdict_match = re.search(r'</Verdict>\s*(.*?)\s*<Verdict>', response, re.DOTALL)
    substitution_match = re.search(r'</Substitution>\s*(.*?)\s*<Substitution>', response, re.DOTALL)
    
    analysis = analysis_match.group(1).strip() if analysis_match else None
    verdict = verdict_match.group(1).strip() if verdict_match else None
    substitution = substitution_match.group(1).strip() if substitution_match else None
    
    return analysis, verdict, substitution

async def process_example(example: Dict, running_id: int, example_id: int, config: BenchmarkConfig) -> List[Dict]:
    """Process a single example using the TutorAgent for analysis with multiple trials"""
    try:
        logger = BenchmarkLogger()
        
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None

        # Extract the expected verdict from data
        expected_verdict = example.get('verdict')
        if not expected_verdict:
            logger.append(f"❌ Warning: No verdict found for example {running_id}")
            logger.append(f"Available fields: {', '.join(example.keys())}")
            logger.print()
            return []

        # Initialize model and agent
        main = get_model(config, role="main")
        tutor = TutorAgent(main)
        
        # Create logs list
        logs = []
        logs.append("\n" + "="*80)
        logs.append(f"📝 Example {running_id + 1} | ID: {example_id}")
        logs.append("="*80)
        logs.append(f"\n📋 Problem:")
        logs.append(f"{example['problem'][:200]}...")
        logs.append(f"\n📝 Solution:")
        logs.append(f"{example['solution'][:200]}...")
        logs.append(f"\n✓ Expected Verdict: {expected_verdict}")
        
        # Run multiple trials
        verdicts = []
        analyses = []
        substitutions = []
        matches = []
        
        for attempt in range(config.best_of):
            
            # Get tutor's analysis
            response = await tutor.find_first_wrong_step(example['problem'], example['solution'])
            # Extract sections from response
            analysis, verdict, substitution = extract_sections(response)
            if verdict:
                verdicts.append(verdict)
                analyses.append(analysis)
                substitutions.append(substitution)
                # Extract just the step number from verdict if it contains "Step"
                verdict_number = None
                if "Step" in verdict:
                    try:
                        verdict_number = verdict.split("Step")[1].split()[0].rstrip('.:)')
                    except:
                        verdict_number = None
                
                # Compare verdicts, allowing for step number match
                verdict_matches = (
                    expected_verdict == verdict or  # Exact match
                    (verdict_number and expected_verdict == verdict_number) or  # Step number match
                    (expected_verdict == "The whole approach is wrong" and "whole approach is wrong" in verdict) or  # Wrong approach match
                    (expected_verdict == "The answer is correct" and "answer is correct" in verdict)  # Correct answer match
                )
                
                matches.append(verdict_matches)
                
                logs.append(f"\n📊 Trial {attempt + 1}:")
                logs.append(f"🤖 Tutor Verdict: {verdict}")
                logs.append(f"✓ Verdict Match: {verdict_matches}")
                
            
        
        if not verdicts:
            logger.append(f"❌ Error: No valid verdicts obtained")
            logger.print()
            return []
        # Calculate statistics
        correct_count = sum(matches)
        success_rate = (correct_count / len(matches)) * 100
        # Find most common verdict
        most_common_verdict = Counter(verdicts).most_common(1)[0][0]
        most_common_correct = expected_verdict == most_common_verdict
        # Create result entries
        results = []
        # Add benchmark data
        # Create partial solution using the most common verdict
        steps = split_into_steps(example['solution'])
        partial_solution = None
        if "Step" in most_common_verdict:
            try:
                wrong_step = int(most_common_verdict.split("Step")[1].split()[0].rstrip('.:)'))
                if wrong_step > 0 and wrong_step <= len(steps):
                    # Join steps up to (but not including) the wrong step
                    partial_solution = "\n".join(steps[:wrong_step-1])
                    # Add the tutor's suggested correction if available
                    if substitutions[0]:  # Use first substitution for simplicity
                        partial_solution += "\n" + substitutions[0]
            except:
                partial_solution = None
        
        results.append({
            'id': example_id,
            'data_type': 'tut_ben',
            'problem': example['problem'],
            'solution': example['solution'],
            'expected_verdict': expected_verdict,
            'tutor_verdicts': verdicts,
            'tutor_analyses': analyses,
            'tutor_substitutions': substitutions,
            'verdict_matches': matches,
            'partial_solution': partial_solution
        })
        
        # Add statistics
        results.append({
            'id': example_id,
            'data_type': 'statistics',
            'example_processed_successfully': True,
            'is_correct_list': matches,
            'is_most_common_correct': most_common_correct,
            'success_rate': success_rate,
            'total_solutions': len(verdicts),
            'correct_solutions': correct_count,
            'incorrect_solutions': len(verdicts) - correct_count,
            'tournament_winner_correct': None,
            'judge_accuracy': None,
            'judge_decisions': 0,
            'all_solutions_correct': all(matches)
        })
        
        # Log statistics
        logs.append("\n📊 Statistics:")
        logs.append(f"├─ Total trials: {len(verdicts)}")
        logs.append(f"├─ Correct verdicts: {correct_count}")
        logs.append(f"├─ Success rate: {success_rate:.1f}%")
        logs.append(f"├─ Most common verdict: {most_common_verdict}")
        logs.append(f"└─ Most common verdict correct? {'Yes' if most_common_correct else 'No'}")
        # Print all logs
        for log in logs:
            logger.append(log)
        logger.print()
        
        return results

    except Exception as e:
        logger = BenchmarkLogger()
        logger.append(f"❌ Error processing example {running_id}: {str(e)}")
        logger.print()
        return []

async def main():
    """Main function for tutor benchmarking approach"""
    config = BenchmarkConfig.from_args('Tutor benchmarking approach')
    
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
