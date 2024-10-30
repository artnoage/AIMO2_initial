import os

def format_text_blocks(text: str, max_line_length: int = 80) -> str:
    """Format text into lines of maximum length while preserving paragraphs"""
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        words = paragraph.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + len(current_line) <= max_line_length:
                current_line.append(word)
                current_length += word_length
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
                
        if current_line:
            lines.append(' '.join(current_line))
            
        formatted_paragraphs.append('\n'.join(lines))
    
    return '\n\n'.join(formatted_paragraphs)

def init_conversation_md(problem_id: str, problem: str, solution: str, solver_model_name: str, suffix: str = ""):
    """Initialize the markdown file with problem details"""
    filename = f"conversation_{problem_id}{suffix}.md"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Create/overwrite the file with initial content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Problem {problem_id} - Solver: {solver_model_name}\n\n")
        f.write("## Problem Statement\n\n")
        f.write(f"{problem}\n\n")
        f.write("## Dataset Solution\n\n")
        f.write(f"{solution}\n\n")
        f.write("## Conversation History\n\n")
    return filename

def append_to_conversation_md(filename: str, role: str, content: str, round_num: int, 
                            messages: str = "", problem: str = ""):
    """Append a new message to the conversation markdown file"""
    if not os.path.exists(filename):
        print(f"Warning: Markdown file {filename} not found")
        return
        
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"### Round {round_num + 1}\n\n")
            
            # Add the prompt that was used
            if role == "Solver's Solution" or role == "Solver 1's Solution":
                prompt = f"""You are a mathematical problem solver. Your goal is to solve this problem:

{problem}

Then solve the problem step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

Never ask for confirmation. Just provide your final answer as a number at the end of your 
response prefixed with 'ANSWER: '.

{messages}"""
                f.write("#### Input Prompt\n")
                f.write("```\n")
                f.write(format_text_blocks(prompt))
                f.write("\n```\n\n")
            elif role == "Solver 2's Solution":
                # Split messages using string operations
                solution_attempt = messages.split("Previous solution: ")[1]
                solution_part = solution_attempt.split("Feedback:")[0].strip()
                feedback_part = messages.split("Feedback: ")[1].strip()
                
                prompt = f"""You are a mathematical problem solver. Here is:

1. The original problem:
{problem}

2. A previous solution attempt:
{solution_part}

3. Feedback on what was wrong:
{feedback_part}

Your task is to fix the solution based on the feedback. Focus specifically on addressing
the issues mentioned in the feedback while keeping the correct parts of the original solution.

Provide your complete revised solution, ending with your final answer prefixed with 'ANSWER: '."""
                f.write("#### Input Prompt\n")
                f.write("```\n")
                f.write(format_text_blocks(prompt))
                f.write("\n```\n\n")
            elif role == "Verifier's Response":
                prompt = f"""You are a mathematical solution verifier. For this problem:

{problem}

The solver's current answer is INCORRECT. Your job is to analyze their solution and try to isolate the most important 
issue with the solution.

Respond with:
'FEEDBACK: [Explanation of errors found and specific suggestions for improvement]'

{messages}"""
                f.write("#### Input Prompt\n")
                f.write("```\n")
                f.write(format_text_blocks(prompt))
                f.write("\n```\n\n")
            
            # Add the agent's response
            f.write(f"#### {role}\n")
            f.write("```\n")
            f.write(format_text_blocks(content))
            f.write("\n```\n\n")
            f.flush()  # Force write to disk
    except Exception as e:
        print(f"Error appending to markdown file: {e}")
