"""The core research agent: a Plan-Execute-Synthesize flow over three tools -
the private-knowledge-base Document Search Tool (phase 1), the Financial Data
Tool (phase 3), and the public-internet Web Search Tool (phase 2) - wrapped
in a phase 4 critique/refinement loop.

ADK conventions: this module exposes `root_agent`, which `adk run src/research_agent`
and `adk web src` discover by name. That symbol is now the `LoopAgent`
(see ADR-0010), not the single-shot planner directly - `research_agent`
below is the renamed former `root_agent`, one sub-agent of the loop.
Plain functions passed via `tools=` are auto-wrapped as function tools, with
their docstrings as tool descriptions; `web_search_tool` is an `AgentTool`
wrapping a sub-agent (see src/tools/web_search.py for why).

Observability: Langfuse tracing is wired in here, not per-entrypoint, since
every entrypoint (`adk run`, `adk web`, the Streamlit UI) imports this module
to get `root_agent` - one instrumentation call covers all three. It must run
before any Agent is constructed (including web_search.py's module-level
sub-agent, and critique.py's module-level `critique_agent`), and after
`src.config` has loaded `.env`, hence the import order below.

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

from google.adk.agents import Agent, LoopAgent
from google.genai import types

from src.tools.document_search import search_documents
from src.tools.financial_data import get_financial_data
from src.tools.web_search import web_search_tool

# Imported after instrument() (see the observability note above) - this
# module's own import constructs critique_agent = Agent(...) at load time.
from src.research_agent.critique import critique_agent, reset_turn_state

INSTRUCTION = """You are a research agent with three sources of evidence: a
private knowledge base (search_documents), live financial market data
(get_financial_data), and the public internet (web_search_tool). For every
question, follow a plan-execute-synthesize flow:

0. Refinement check: {critique_followups?} holds specific follow-up
   sub-questions a prior critique pass raised against your last draft for
   this same turn (empty if this is the first pass, which is the common
   case). If it is non-empty, treat each one as an additional fact to plan
   and execute for, on top of - not instead of - the original question,
   and produce a new complete draft that folds the answer to each follow-up
   into the previous draft rather than just appending to it.
1. Plan: break the question into the distinct facts you need. For each, decide
   which single source is appropriate - search_documents for anything about
   the private knowledge base's own documents; get_financial_data for current
   prices or movements of stocks, cryptocurrencies, or currency exchange
   rates (never use web_search_tool for those - the financial tool's
   predefined sources are the authority on them); web_search_tool for
   anything else public, current, or outside those documents. Only plan to
   use multiple sources for a fact if the question genuinely requires
   combining evidence across them - not as a routine double-check of a
   source that already answers the fact on its own. State the plan briefly.
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
   source of each fact. A citation must identify something a reader could go
   and check, and each tool gives you one - use what it gives you rather than
   naming the tool itself. Never write "(Google Search)", "(web search)" or
   any other tool name as a source; that is not a citation.
   - search_documents: cite the document name and, where the passage gives
     one, the page.
   - web_search_tool: its result ends with a "Sources:" list of domains and
     URLs. Cite the URL of the source a fact came from.
   - get_financial_data: its result includes a "source" field holding the URL
     the figures were fetched from. Cite that URL.
   If sources
   conflict, say so explicitly rather than silently picking one - prefer the
   private knowledge base as authoritative for anything the knowledge base
   itself covers, and note the discrepancy. If neither source contains the
   specific answer asked for, say so plainly and stop there - do not
   substitute related-but-different facts as if they were the answer, even
   framed as "additional context". State each caveat once; never repeat a
   sentence, disclaimer, or phrase.
"""

# A caveat once is enough (see the synthesize step) - but the last line of
# defence against a caveat repeating anyway is generation config, not prompt
# wording: without a max_output_tokens cap, a Gemini decoding loop can run
# to the model's own hard ceiling. Observed once via Langfuse trace on a
# real turn - "of course, this is a simulated real-time market rate..."
# repeated 2515 times, 65,532 output tokens, 4m35s - before this cap
# existed. frequency_penalty targets the same failure mode more directly,
# making each repeat of an already-used token increasingly costly.
_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    max_output_tokens=4096,
    frequency_penalty=0.4,
)

# Renamed from root_agent (see ADR-0010): this is now one sub-agent of the
# critique loop below, not the module's discovered entrypoint. output_key
# writes its final answer to session state as "draft_answer" - the only
# thing critique_agent is allowed to see of this cycle's work (see
# src/research_agent/critique.py's module docstring for why).
research_agent = Agent(
    name="research_agent",
    model=config.GEMINI_MODEL,
    description="Answers questions over a private knowledge base, live financial market data, and the public internet via planned, multi-source search.",
    instruction=INSTRUCTION,
    tools=[search_documents, get_financial_data, web_search_tool],
    generate_content_config=_GENERATE_CONTENT_CONFIG,
    output_key="draft_answer",
)

# The phase 4 critique/refinement loop (ADR-0010). LoopAgent is deprecated in
# google-adk 2.5.0 in favour of Workflow, but Workflow cannot yet be used as
# an LlmAgent sub-agent, which today's composition (research_agent and
# critique_agent are both LlmAgents) requires - so LoopAgent is the
# deliberate choice, not an oversight; revisit when that Workflow limitation
# lifts. max_iterations is the hard, code-level ceiling (ADR-0010's "two-tier
# iteration bound") - no request can raise it. The per-request soft budget
# that most turns actually stop on lives in session state
# ("critique_budget"), read by critique_agent's before_agent_callback, not
# here.
#
# root_agent is now this LoopAgent, not research_agent directly - `adk run`/
# `adk web` discover the agent to run by that exact module-level name, so it
# has to stay attached to whichever object is the actual entrypoint.
#
# before_agent_callback=reset_turn_state resets the loop's per-turn state
# (how much of the budget this turn has spent so far, the original
# question, any leftover follow-ups) exactly once, before the first cycle -
# see critique.py's reset_turn_state docstring. Without it, a session that
# spans multiple user turns (both scripts/verify_agent.py and the Streamlit
# UI reuse one session that way) would leak one turn's critique bookkeeping
# into the next, unrelated turn.
root_agent = LoopAgent(
    name="research_loop",
    description="Runs the research agent, critiques its draft, and either ends the turn or feeds follow-up questions back for another research cycle.",
    sub_agents=[research_agent, critique_agent],
    max_iterations=config.MAX_CRITIQUE_ITERATIONS,
    before_agent_callback=reset_turn_state,
)
