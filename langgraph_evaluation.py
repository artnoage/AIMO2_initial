import os
from functools import partial
from typing import Annotated, TypedDict, Union, List
from datasets import load_dataset
from langgraph.graph.message import add_messages
import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import print_graph
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate

from enum import Enum, auto

# Setup for OpenRouter
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

class ModelOption(Enum):
    CLAUDE_3_SONNET = "anthropic/claude-3.5-sonnet:beta"
    GEMINI_PRO = "google/gemini-pro-1.5"
    GPT4 = "openai/gpt-4o"

# Default models
SOLVER_MODEL = ModelOption.CLAUDE_3_SONNET
VERIFIER_MODEL = ModelOption.CLAUDE_3_SONNET

# Define state schema
class AgentState(TypedDict):
    solver_messages: Annotated[List[BaseMessage], add_messages]
    verifier_messages: Annotated[List[BaseMessage], add_messages]
    current_solution: Annotated[str, "Current solution being worked on"]
    final_answer: Annotated[Union[int, None], "Final numerical answer"]
    iteration_count: Annotated[int, "Counter for solver-verifier iterations"]

# Initialize the models
def get_model(model: ModelOption, temp: float = 0):
    return ChatOpenAI(
        model=model.value,
        temperature=temp,
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

# Define system prompts
SOLVER_PROMPT = """You are a mathematical problem solver. Here is the problem:

{problem}

Previous messages:
{messages}

Before solving, start with a brief analysis:
1. What mathematical concepts is this problem testing?
2. What theoretical tools or formulas might be useful?
3. Are there any tricks or simplifications that could make this problem easier?

Then solve the problem step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for bravity.
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it. 

Provide your final answer as a number at the end of your response prefixed with 'ANSWER: '."""

VERIFIER_PROMPT = """You are a mathematical solution verifier. For this problem:

{problem}

Previous messages:
{messages}

Your job is to rigorously verify the solver's solution. Follow these steps:

1. Check the initial approach:
   - Is the chosen method appropriate?
   - Are all necessary concepts being used correctly?

2. Verify calculations:
    If possible, work backwards from the answer to verify it satisfies the initial conditions, 
    or the intermediate steps. 
   
3. Examine logic:
   - Are all steps properly justified?
   - Are there any gaps in reasoning?
   - Are edge cases considered?

4. Validate the final answer:
   - Compare with the known correct answer
   - Verify all calculations leading to the answer
   - Check if the reasoning is sound even if answer is correct
   - Ensure units and context are appropriate

If the answer matches the ground truth AND the reasoning is correct, respond with:
'VERIFIED: [Brief explanation of what you checked and why it's correct]'

If the answer is wrong OR the reasoning has issues, respond with:
'NEEDS_REVISION: [Specific feedback about where the solution went wrong, without revealing the correct answer. DO NOT LEAK THE GROUND TRUTH TO THEY SOLVER.
Point out specific steps or assumptions that need review. For example: "The approach in step 2 is problematic because..." or "The calculation in step 3 doesn't follow from step 2 because..."]'"""

# Create the chains
def preprocess_template_vars(text: str) -> str:
    """Replace any template-like variables in the text with escaped versions"""
    # First escape any existing curly braces that aren't template variables
    processed_text = text.replace("{{", "{{{{").replace("}}", "}}}}")
    
    # Find all template variables like {ABC} or {n}
    var_pattern = r'\{([^{}]+)\}'
    template_vars = set(re.findall(var_pattern, processed_text))
    
    # Create a dictionary of replacements
    replacements = {}
    for var in template_vars:
        if var == 'messages' or var == 'problem':  # Keep special variables
            continue
        replacements[f"{{{var}}}"] = f"{{{{n}}}}" if var == 'n' else f"{{var_{var}}}"
    
    # Apply replacements
    for old, new in replacements.items():
        processed_text = processed_text.replace(old, new)
    
    return processed_text

def create_solver_chain(problem: str, model_option: ModelOption = SOLVER_MODEL):
    model = get_model(model_option)
    # First escape any literal curly braces
    escaped_problem = problem.replace("{", "{{").replace("}", "}}")
    
    # Create the prompt with the escaped problem text
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOLVER_PROMPT.format(problem=escaped_problem, messages="{messages}"))
    ])
    return prompt | model

def create_verifier_chain(problem: str, model_option: ModelOption = VERIFIER_MODEL):
    model = get_model(model_option, temp=0.1)
    # First escape any literal curly braces
    escaped_problem = problem.replace("{", "{{").replace("}", "}}")
    
    # Create the prompt with the escaped problem text
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFIER_PROMPT.format(
            problem=escaped_problem,
            messages="{messages}")
        )
    ])
    return prompt | model

def solve(state: AgentState, solver_chain):
    """Solver agent function"""
    # Convert messages list to string
    messages_text = "\n".join([msg.content for msg in state["solver_messages"]])
    
    # Get response from the solver
    response = solver_chain.invoke({"messages": messages_text})
    solution_content = response.content
    
    # Create both AI and Human versions of the response
    ai_message = AIMessage(content=solution_content)
    human_message = HumanMessage(content=solution_content)
    
    return {
        "current_solution": solution_content,
        "solver_messages": [ai_message],
        "verifier_messages": [human_message]
    }

def verify(state: AgentState, verifier_chain):
    """Verifier agent function"""
    
    # Convert messages list to string
    messages_text = "\n".join([msg.content for msg in state["verifier_messages"]])
    
    # Get response from the verifier
    response = verifier_chain.invoke({"messages": messages_text})

    
    # Create both AI and Human versions of the response
    ai_message = AIMessage(content=response.content)  # Keep original for verifier
    human_message = HumanMessage(content=response.content)  # Censored for solver
    
    return {
        "solver_messages": [human_message],
        "verifier_messages": [ai_message],
        "iteration_count": state["iteration_count"] + 1
    }

def check_answer(state: AgentState, ground_truth: int) -> Union[str, None]:
    """Check if the answer is correct and determine next step"""
    if state["final_answer"] == ground_truth or state["iteration_count"] >= 3:
        return END
    
    print("\n🔄 Verifier was wrong - resampling verifier...\n")
    
    # Remove last messages and try verification again
    if len(state["solver_messages"]) > 0:
        state["solver_messages"].pop()
    if len(state["verifier_messages"]) > 0:
        state["verifier_messages"].pop()
    
    return "verifier"

def should_continue(state: AgentState) -> str:
    """Determine if we need another iteration"""
    last_message = state["verifier_messages"][-1].content
    if "NEEDS_REVISION" in last_message:
        return "solver"
    return "cleaner"

def clean_answer(state: AgentState) -> AgentState:
    """Extract numerical answer from verifier's response"""
    last_message = state["verifier_messages"][-1].content
    if "VERIFIED" in last_message:
        # Try to extract a number from the solver's solution
        solution = state["current_solution"]
        match = re.search(r"ANSWER:\s*(\d+)", solution)
        if match:
            final_answer = int(match.group(1))
        else:
            # Fallback to any number in the solution
            match = re.search(r"\d+", solution)
            final_answer = int(match.group()) if match else None
    else:
        final_answer = None
    
    # Note: This call will be handled in the main loop instead
    pass
    
    return {
        **state,
        "final_answer": final_answer
    }

def save_conversation_to_md(state: AgentState, problem_id: str, problem: str, solution: str):
    """Save the conversation to a Markdown file"""
    filename = f"conversation_{problem_id}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Problem {problem_id}\n\n")
        f.write("## Problem Statement\n\n")
        f.write(f"{problem}\n\n")
        f.write("## Dataset Solution\n\n")
        f.write(f"{solution}\n\n")
        f.write("## Conversation History\n\n")
        messages = []
        for i in range(len(state["solver_messages"])):
            if i % 2 == 0:
                messages.append(("Solver's Solution", state["solver_messages"][i].content))
            else:
                messages.append(("Verifier's Response", state["solver_messages"][i].content))
                
        for i, (role, content) in enumerate(messages, 1):
            f.write(f"### Round {(i + 1) // 2}\n\n")
            f.write(f"#### {role}\n")
            f.write("```\n")
            f.write(f"{content}\n")
            f.write("```\n\n")

def build_graph(solver_chain, verifier_chain, ground_truth: int):
    """Build the workflow graph for a specific problem"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("solver", partial(solve, solver_chain=solver_chain))
    workflow.add_node("verifier", partial(verify, verifier_chain=verifier_chain))
    workflow.add_node("cleaner", clean_answer)
    workflow.add_node("checker", partial(check_answer, ground_truth=ground_truth))

    # Add edges
    workflow.set_entry_point("solver")
    workflow.add_edge("solver", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        should_continue,
        {
            "solver": "solver",
            "cleaner": "cleaner"
        }
    )
    workflow.add_edge("cleaner", "checker")
    workflow.add_conditional_edges(
        "checker",
        lambda x: x,
        {
            "verifier": "verifier",
            END: END
        }
    )

    return workflow

def process_problem(problem_text: str, ground_truth: int, 
                   solver_model: ModelOption = SOLVER_MODEL,
                   verifier_model: ModelOption = VERIFIER_MODEL):
    """Process a single problem through the graph"""
    # Create chains for this specific problem
    solver_chain = create_solver_chain(problem_text, solver_model)
    verifier_chain = create_verifier_chain(problem_text, verifier_model)
    
    # Initialize state
    initial_state = {
        "solver_messages": [],
        "verifier_messages": [],
        "current_solution": "",
        "final_answer": None,
        "iteration_count": 0
    }
    
    # Build and compile graph for this problem
    workflow = build_graph(solver_chain, verifier_chain, ground_truth)
    
    # Print the graph visualization
    print("\nWorkflow Graph Structure:")
    try:
        display(Image(workflow.get_graph().draw_mermaid_png()))
    except Exception:
        # Fallback to basic print if visualization fails
        print_graph(workflow)
    
    app = workflow.compile()
    
    # Run the graph
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    # Load first 3 problems from AIME dataset
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train[3:8]")
    
    results = []
    for example in dataset:
        problem_id = example['id']
        problem = example['problem']
        print(f"\nProcessing problem {problem_id}...")
        
        ground_truth = int(example['answer']) if example['answer'].isdigit() else None
        result = process_problem(problem, ground_truth)
        results.append({
            'problem_id': problem_id,
            'final_answer': result['final_answer'],
            'ground_truth': int(example['answer']) if example['answer'].isdigit() else None
        })
        
        # Save conversation after processing
        save_conversation_to_md(result, problem_id, example['problem'], example['solution'])
    
    # Print summary
    print("\nResults Summary:")
    for result in results:
        print(f"\nProblem {result['problem_id']}:")
        print(f"Model Answer: {result['final_answer']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Correct: {result['final_answer'] == result['ground_truth']}")
