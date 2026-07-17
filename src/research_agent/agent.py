"""The core RAG agent (phase 1): a basic Plan-Execute-Synthesize flow over the
Document Search Tool.

ADK conventions: this module exposes `root_agent`, which `adk run src/research_agent`
and `adk web src` discover by name. Plain functions passed via `tools=` are
auto-wrapped as function tools, with their docstrings as tool descriptions.

NOTE: written against the ADK docs (https://google.github.io/adk-docs/) before
the dependency was installed - treat as a skeleton to verify against the real
API on first `adk run`, not as tested code.
"""

from google.adk.agents import Agent

from src import config
from src.tools.document_search import search_documents

INSTRUCTION = """You are a research agent answering questions from a private
knowledge base. For every question, follow a plan-execute-synthesize flow:

1. Plan: break the question into the distinct facts you need, and decide what
   to search for. State the plan briefly.
2. Execute: call search_documents for each planned search. Reformulate and
   search again if the first results do not contain what you need.
3. Synthesize: answer strictly from the retrieved passages, citing the source
   document of each fact. If the knowledge base does not contain the answer,
   say so plainly instead of guessing.
"""

root_agent = Agent(
    name="research_agent",
    model=config.GEMINI_MODEL,
    description="Answers questions over a private knowledge base via planned document search.",
    instruction=INSTRUCTION,
    tools=[search_documents],
)
