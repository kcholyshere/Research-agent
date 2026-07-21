"""The core research agent: a Plan-Execute-Synthesize flow over two tools -
the private-knowledge-base Document Search Tool (phase 1) and the public-
internet Web Search Tool (phase 2).

ADK conventions: this module exposes `root_agent`, which `adk run src/research_agent`
and `adk web src` discover by name. Plain functions passed via `tools=` are
auto-wrapped as function tools, with their docstrings as tool descriptions;
`web_search_tool` is an `AgentTool` wrapping a sub-agent (see
src/tools/web_search.py for why).

NOTE: written against the ADK docs (https://google.github.io/adk-docs/) before
the dependency was installed - treat as a skeleton to verify against the real
API on first `adk run`, not as tested code.

Observability: Langfuse tracing is wired in here, not per-entrypoint, since
every entrypoint (`adk run`, `adk web`, the Streamlit UI) imports this module
to get `root_agent` - one instrumentation call covers all three. It must run
before any Agent is constructed (including web_search.py's module-level
sub-agent), and after `src.config` has loaded `.env`, hence the import order
below.
"""

from src import config  # noqa: F401 - import first: triggers .env load via dotenv

from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from langfuse import get_client

GoogleADKInstrumentor().instrument()
langfuse_client = get_client()

from google.adk.agents import Agent

from src.tools.document_search import search_documents
from src.tools.web_search import web_search_tool

INSTRUCTION = """You are a research agent with two sources of evidence: a
private knowledge base (search_documents) and the public internet
(web_search_tool). For every question, follow a plan-execute-synthesize flow:

1. Plan: break the question into the distinct facts you need. For each, decide
   which source is appropriate - search_documents for anything about the
   private knowledge base's own documents, web_search_tool for anything
   public, current, or outside those documents. Use both when a question
   needs combining private-document facts with public context. State the
   plan briefly.
2. Execute: call the planned tool(s) for each fact. Reformulate and search
   again if the first results do not contain what you need.
3. Synthesize: answer strictly from the retrieved passages/results, citing the
   source (document name, or URL for web results) of each fact. If sources
   conflict, say so explicitly rather than silently picking one - prefer the
   private knowledge base as authoritative for anything the knowledge base
   itself covers, and note the discrepancy. If neither source contains the
   answer, say so plainly instead of guessing.
"""

root_agent = Agent(
    name="research_agent",
    model=config.GEMINI_MODEL,
    description="Answers questions over a private knowledge base and the public internet via planned, multi-source search.",
    instruction=INSTRUCTION,
    tools=[search_documents, web_search_tool],
)
