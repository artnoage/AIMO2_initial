import os
from typing import Annotated, Sequence, TypedDict, Union
from typing_extensions import TypeVar
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, MessageGraph
from langgraph.prebuilt.message_graphs import ChatGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Setup for OpenRouter
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
# Both agents using claude-3-sonnet for now
SOLVER_MODEL = "anthropic/claude-3-sonnet"
VERIFIER_MODEL = "anthropic/claude-3-sonnet"

# Define state schema
class AgentState(TypedDict):
    solver_messages: Annotated[Sequence[BaseMessage], "Messages for the solver"]
    verifier_messages: Annotated[Sequence[BaseMessage], "Messages for the verifier"]
    current_solution: Annotated[str, "Current solution being worked on"]
    all_messages: Annotated[Sequence[BaseMessage], "All messages in the conversation"]

# Initialize the models
def get_model(model_name: str):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

# Setup solver agent
solver = ChatPromptTemplate.from_messages([
    ("system", "You are a mathematical problem solver. Your task is to solve math problems step by step, "
              "showing your work clearly. Provide your final answer as a number at the end of your response "
              "prefixed with 'ANSWER: '. Be thorough but concise."),
    MessagesPlaceholder(variable_name="solver_messages"),
    ("human", "{input}")
])

# Setup verifier agent
verifier = ChatPromptTemplate.from_messages([
    ("system", "You are a mathematical solution verifier. Your job is to check if the solver's solution is correct. "
              "Look for any errors in logic or calculation. Respond with either:\n"
              "'VERIFIED: [explanation]' if the solution looks correct\n"
              "'NEEDS_REVISION: [explanation]' if you find any issues"),
    MessagesPlaceholder(variable_name="verifier_messages"),
    ("human", "Please verify this solution:\n{solution}")
])

def solve(state: AgentState, model: ChatOpenAI = get_model(SOLVER_MODEL)):
    """Solver agent function"""
    prompt = solver.invoke({
        "solver_messages": state["solver_messages"],
        "input": state["all_messages"][-1].content if state["all_messages"] else "No input provided"
    })
    response = model.invoke(prompt)
    return {
        "solver_messages": [*state["solver_messages"], response],
        "current_solution": response.content,
        "all_messages": [*state["all_messages"], response]
    }

def verify(state: AgentState, model: ChatOpenAI = get_model(VERIFIER_MODEL)):
    """Verifier agent function"""
    prompt = verifier.invoke({
        "verifier_messages": state["verifier_messages"],
        "solution": state["current_solution"]
    })
    response = model.invoke(prompt)
    return {
        "verifier_messages": [*state["verifier_messages"], response],
        "all_messages": [*state["all_messages"], response]
    }

def should_continue(state: AgentState) -> Union[str, None]:
    """Determine if we need another iteration"""
    last_message = state["all_messages"][-1].content
    if "NEEDS_REVISION" in last_message:
        return "solver"
    return "end"

# Build the graph
workflow = Graph()

# Add nodes
workflow.add_node("solver", solve)
workflow.add_node("verifier", verify)

# Add edges
workflow.add_edge("solver", "verifier")
workflow.add_conditional_edges(
    "verifier",
    should_continue,
    {
        "solver": "solver",
        "end": None
    }
)

# Compile the graph
app = workflow.compile()

def process_problem(problem_text: str):
    """Process a single problem through the graph"""
    # Initialize state
    initial_state = {
        "solver_messages": [],
        "verifier_messages": [],
        "current_solution": "",
        "all_messages": [HumanMessage(content=problem_text)]
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    # Example usage
    problem = "If a^2 + b^2 = 25 and ab = 12, find the value of (a+b)^2."
    result = process_problem(problem)
    print("\nFinal conversation:")
    for msg in result["all_messages"]:
        print(f"\n{'Bot' if isinstance(msg, AIMessage) else 'Human'}: {msg.content}")
