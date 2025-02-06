import os
import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.progress_tracker import ProgressTracker
from utils.benchmark_utils import get_model, extract_answer_from_solution
from utils.agents import TutorAgent
from utils.logger import BenchmarkLogger

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
    """Process a single example using the TutorAgent for analysis"""
    try:
        logger = BenchmarkLogger()
        
        if not isinstance(example, dict) or 'problem' not in example or 'solution' not in example:
            logger.append(f"❌ Error processing example {str(running_id)}: Invalid example format")
            logger.print()
            return None

        # Extract the expected verdict from auxiliary data
        expected_verdict = example.get('verdict', None)
        if not expected_verdict:
            logger.append(f"❌ Warning: No verdict found for example {running_id}")
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
        
        # Get tutor's analysis
        response = await tutor.find_first_wrong_step(example['problem'], example['solution'])
        
        # Extract sections from response
        analysis, verdict, substitution = extract_sections(response)
        
        if not verdict:
            logger.append(f"❌ Error: Could not extract verdict from response")
            logger.print()
            return []
            
        # Create result entry
        result = {
            'id': example_id,
            'data_type': 'benchmark',
            'problem': example['problem'],
            'solution': example['solution'],
            'expected_verdict': expected_verdict,
            'tutor_verdict': verdict,
            'tutor_analysis': analysis,
            'tutor_substitution': substitution,
            'verdict_match': expected_verdict == verdict
        }
        
        # Log results
        for log in logs:
            logger.append(log)
        logger.append(f"\n🤖 Tutor Verdict: {verdict}")
        logger.append(f"✓ Verdict Match: {result['verdict_match']}")
        
        logger.print()
        return [result]

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
