import os
import asyncio
import logging
from typing import Dict, List
from dotenv import load_dotenv
from utils.benchmark_config import BenchmarkConfig
from utils.benchmark_utils import get_model
from utils.logger import BenchmarkLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
load_dotenv()

QUESTIONS = [
    # Math questions
    "What is the derivative of x^2 with respect to x?",
    "What is the integral of 2x dx?",
    "Solve the equation: x^2 - 4 = 0",
    "What is the value of sin(π/2)?",
    
    # General knowledge questions
    "What are three main differences between renewable and non-renewable energy sources?",
    "Explain how the water cycle works in simple terms.",
    "What makes a good scientific hypothesis?",
    "How does the human immune system protect against diseases?"
]

async def ask_questions(model) -> List[Dict]:
    """Ask predefined questions and get responses"""
    logger = BenchmarkLogger()
    results = []
    
    for i, question in enumerate(QUESTIONS, 1):
        try:
            logger.append(f"\n📝 Question {i}:")
            logger.append(question)
            
            response = await model.ainvoke(question)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.append(f"\n🤖 Response:")
            logger.append(response_text)
            logger.append("\n" + "="*50)
            
            results.append({
                'question': question,
                'response': response_text.strip() if response_text else "Error: No response content"
            })
            
        except Exception as e:
            logger.append(f"❌ Error getting response: {str(e)}")
    
    logger.print()
    return results

async def main():
    """Main function for asking simple questions"""
    config = BenchmarkConfig.from_args('Simple questions to LLM')
    
    # Initialize model
    main = get_model(config, role="main")
    
    # Ask questions and get responses
    results = await ask_questions(main)
    
    # Save results to file
    output_dir = "simple_questions_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create markdown output
    markdown_content = ["# Simple Questions and Answers\n"]
    for i, result in enumerate(results, 1):
        markdown_content.extend([
            f"## Question {i}",
            result['question'],
            "\n### Answer",
            result['response'],
            "\n---\n"
        ])
    
    # Write to file
    with open(f"{output_dir}/qa_results.md", "w") as f:
        f.write("\n".join(markdown_content))
    
    print(f"\n✓ Results saved to {output_dir}/qa_results.md")

if __name__ == "__main__":
    logger = BenchmarkLogger()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.append("\n❌ Questions interrupted by user")
        logger.print()
    except Exception as e:
        logger.append(f"\n❌ Questions failed with error: {e}")
        logger.print()
