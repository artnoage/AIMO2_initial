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

def init_conversation_md(problem_id: str, problem: str, solution: str, solver_model_name: str, suffix: str = "", directory: str = ""):
    """Initialize the markdown file with problem details"""
    # Create solver-specific subfolder
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"conversation_{problem_id}{suffix}.md")
    
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
                            prompt: str = ""):
    """Append a new message to the conversation markdown file"""
    if not os.path.exists(filename):
        print(f"Warning: Markdown file {filename} not found")
        return
        
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"### Round {round_num + 1}\n\n")
            
            # Add the prompt if provided
            if prompt:
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
