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

Repo-root sys.path bootstrap: ADK's CLI loader (`adk run`/`adk web`) inserts
only the agent's *parent* directory (`src`) onto sys.path, not the repo root,
so a bare `from src import config` fails under that entrypoint even though it
works under `python -m` and Streamlit (both of which put the repo root on
sys.path via cwd). Fixing it up here, from this file's own location, makes
`from src import config` work under every entrypoint without relying on the
caller to invoke things a particular way.
"""

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
   which single source is appropriate - search_documents for anything about
   the private knowledge base's own documents, web_search_tool for anything
   public, current, or outside those documents. Only plan to use both sources
   for a fact if the question genuinely requires combining private-document
   evidence with public context - not as a routine double-check of a source
   that already answers the fact on its own. State the plan briefly.
2. Execute: call only the tool(s) you planned for each fact, once each. If a
   result already contains the fact you planned it for, that fact is done -
   never issue another search to "verify", "confirm", or add detail beyond
   what was asked. Tool results come from the live web and the current
   knowledge base, which are more up to date than your training data: trust
   them over your own sense of what has or hasn't happened yet, and never
   search to check today's date or to double-check a result that surprised
   you. Reformulate and search again on the same source only if the first
   results do not contain what you need; only fall back to the other source
   if the fact's own planned source turns out not to cover it.
3. Synthesize: answer strictly from the retrieved passages/results, citing the
   source (document name, or URL for web results) of each fact. If sources
   conflict, say so explicitly rather than silently picking one - prefer the
   private knowledge base as authoritative for anything the knowledge base
   itself covers, and note the discrepancy. If neither source contains the
   specific answer asked for, say so plainly and stop there - do not
   substitute related-but-different facts as if they were the answer, even
   framed as "additional context".
"""

root_agent = Agent(
    name="research_agent",
    model=config.GEMINI_MODEL,
    description="Answers questions over a private knowledge base and the public internet via planned, multi-source search.",
    instruction=INSTRUCTION,
    tools=[search_documents, web_search_tool],
)
