import re
import os
import asyncio
import argparse
from functools import partial
from typing import Annotated, TypedDict, Union, List, Dict, Optional
from utils.utils import extract_answer_from_solution
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from utils.librarian import init_conversation_md, append_to_conversation_md
import tiktoken
from utils.utils import ModelOption, get_model


# Load environment variables from .env
load_dotenv()
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# Define system prompts as constants
SOLVER_PROMPT_TEMPLATE = """You are a mathematical problem solver. When given a problem, solve it step by step, showing your work clearly. Make sure to:
- Explain your reasoning at each step
- Show all calculations explicitly
- Never omit calculations for brevity
- Highlight any key insights or clever observations
- If some calculations seem hard, think if there is a clever way around it

In the end provide your final answer inside \\boxed{}"""

JUDGE_PROMPT_TEMPLATE = """You are a mathematical solution judge. You will be given multiple different solutions to the same problem. Your task is to:
1. Review all solutions carefully
2. Identify the most common answer if there is one
3. Evaluate the reasoning in solutions that arrived at this answer
4. Make a final determination of the most likely correct answer

Provide your final answer inside \\boxed{}"""


