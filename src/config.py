import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vertex AI auth (no API keys - relies on Application Default Credentials).
# ADK reads GOOGLE_GENAI_USE_VERTEXAI/GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION
# from the environment itself; these mirrors are for our own modules.
#
# Deliberately not validated here (audit finding 15a): this module is
# imported unconditionally, including by code and tests that never touch
# Vertex, so failing at import time on a blank value would break hermetic,
# credential-free test runs that have no reason to care. The real check
# lives in src/services/genai_client.get_client() - the actual point a
# genai.Client gets built - see that function's comment for why a blank
# project is a silent misrouting risk rather than a loud one otherwise.
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# Models (phase 1 requirements specify Gemini 2.0 Flash; retired from
# gd-gcp-internship-ds's Vertex AI catalogue by the time of verification -
# see ADR-0004 for the substitution)
GEMINI_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Knowledge base source documents (phase 1 corpus: IFC's 2024 annual report,
# see ADR-0001 - same file Finrag used, dropped as-is into data/raw/)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Chunking (carried over from Finrag's tuned values)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Largest table chunk, in characters, before it is split into several
# (src/ingestion/chunk.py). Tables were kept whole until 2026-08-07 on the
# reasoning that a table is already a coherent unit, which is true of meaning
# and false of embeddings: the audit's finding 10 measured 14 table chunks
# over 8,000 characters and the largest at 31,590, against an embedding input
# limit around 2,048 tokens. The docstore keeps the full text either way, so
# nothing was ever answered wrongly - the tail of an oversized table simply
# became unfindable by similarity search, silently.
#
# 6,000 rather than a token count because no offline tokeniser for
# gemini-embedding-001 is available to measure against, so this is set
# conservatively and then VERIFIED empirically: `python -m src.dataset` prints
# a truncation report, and that report reading zero is the check that this
# number is low enough. Raise it only against that report, never by argument.
TABLE_CHUNK_MAX_CHARS = 6000

# Vector store (phase 1 requirement: FAISS, in-memory document search)
FAISS_INDEX_DIR = PROJECT_ROOT / "models" / "faiss"

# Retrieval
TOP_K = 4

# Phase 4 critique loop (ADR-0010): two separate bounds, on purpose. This one
# is the hard ceiling - it lives in code, not a request field, so no request
# can ever raise it. 3 gives enough headroom to demonstrate more than one
# refinement cycle without letting worst-case latency multiply unboundedly -
# ADR-0009 already had to add a config-level cap once, for the same reason a
# prompt alone can't bound a worst case.
MAX_CRITIQUE_ITERATIONS = 3

# The soft, per-request default (see agent_docs/decisions.md ADR-0010): one
# critique pass is enough to demonstrate the requirement while keeping
# typical latency close to the pre-phase-4 baseline. A request-supplied
# budget of 0 reproduces that baseline exactly - no critique LLM call at all.
DEFAULT_CRITIQUE_BUDGET = 1

# web_search_agent's thinking budget (src/tools/web_search.py), in tokens -
# 0 disables thinking, -1 is Gemini's own "automatic" budget. Measured
# against a direct Vertex probe (n=6): unset (automatic) put this sub-agent
# at a median 19.95s/3,310 thinking tokens per call; 512 measured 9.46s for
# the same probe. 512 ("Medium" in the Streamlit UI) is the default; the UI
# also offers Off/Low/High, settable per request the same way
# DEFAULT_CRITIQUE_BUDGET is.
DEFAULT_WEB_SEARCH_THINKING_BUDGET = 512

# Cumulative token ceiling for one session (src/research_agent/token_budget.py).
# The third bound in this project and the only session-scoped one:
# max_output_tokens caps a response (ADR-0009), MAX_TOOL_CALLS_PER_TURN caps a
# turn's searching (ADR-0015), and neither caps a conversation.
#
# 200,000 comes from measurement, not from the model's context window. Two
# plain knowledge-base questions in one session cost about 16.9k and 18.5k
# tokens (2026-08-06) - and the second is dearer than the first only because
# every turn re-sends the transcript, so per-turn cost climbs for the whole
# session. 200,000 is therefore roughly 8-12 real turns rather than the ~11
# that dividing by the first turn would suggest. Generous for the demo
# sessions this project actually runs, and low enough that an unattended chat
# left open cannot run indefinitely.
MAX_SESSION_TOKENS = 200_000

# Conversation history window for research_agent (src/research_agent/history_trim.py,
# the ADR-0021 follow-up TODOS.md named: "trim conversation history to the last
# N turns"). Bounds each turn's OWN cost, which MAX_SESSION_TOKENS does not -
# that caps the sum across a session, so it says nothing about how expensive
# turn 8 is compared to turn 2. Every model call resends the full transcript,
# and ADR-0021 measured that transcript costing ~7.1k tokens of resent history
# just from turn 1 by the time turn 2 starts.
#
# 3, from a direct measurement (scripts/verify_history_trim.py) of the same
# request, before and after the trim, inside one session - not a comparison
# across two separate live runs, which turned out to be too noisy to trust:
# an earlier draft compared prompt_token_count across a trimmed and an
# untrimmed session and got DIFFERENT numbers on turns neither session had
# reason to differ on yet, because each session's own live web_search_agent
# calls return their own live grounding results. Counting
# llm_request.contents (via the model's own count_tokens) immediately either
# side of the trim callback removes that source of noise entirely: on one
# real run, turn 5's first call dropped from 31 to 24 contents and 10,226 to
# 8,749 tokens - a real 1,477 tokens off that one request, for turns 1-4
# untouched.
#
# The per-turn cost is too tool-dependent for a fixed per-turn TOKEN budget
# to be the unit (a search_documents turn is cheap, a web_search_agent
# turn's grounded results are not - the same run's turns ranged from ~200 to
# ~3,000 tokens each), which is why this bounds the COUNT of retained turns
# instead. 3 keeps every later turn's resent history to at most 3 turns'
# worth rather than all of them, while still covering the common real
# follow-up ("what about the year before?" references the turn immediately
# prior) with headroom for one or two hops further back before a question
# has to be re-searched instead of read from history. The turn in progress
# is never counted against this - see history_trim.py.
MAX_HISTORY_TURNS = 3

# News Agent service (phase 5, Agent-to-Agent demo) - a separate process
# reached over plain HTTP, not an in-process import; see
# src/news_service/server.py and src/tools/news_agent.py for why. A
# dedicated timeout, not genai_client.HTTP_TIMEOUT_MS/MODEL_CALL_TIMEOUT_MS
# (those bound Vertex calls this tool never makes directly): this bounds
# the HTTP hop to a local service that the tool must degrade around rather
# than hang on - see that tool's docstring.
NEWS_AGENT_URL = os.getenv("NEWS_AGENT_URL", "http://localhost:8001")
NEWS_AGENT_TIMEOUT_S = 20.0

# MCP fetch server (phase 3, Financial Data Tool) - the reference `fetch`
# server fronted by a stdio-to-HTTP proxy so it can be an ordinary compose
# service rather than a container the agent spawns per call (ADR-0020).
# The default is the published port of the `mcp-fetch` compose service, which
# is what a local checkout talks to; inside the compose network the agent
# overrides this with the service name.
MCP_FETCH_URL = os.getenv("MCP_FETCH_URL", "http://localhost:8090/mcp")

# Bounds the HTTP hop to that service, for the same reason NEWS_AGENT_TIMEOUT_S
# bounds the A2A hop: neither is a Vertex call, so genai_client's model-call
# timeouts never apply to them. Generous relative to the News Agent's 20s
# because the server's own work is a live page fetch of a third-party site.
MCP_FETCH_TIMEOUT_S = 30.0

# Wall-clock ceiling on a whole turn (src/research_agent/turn_deadline.py) -
# the bound TODOS.md flagged as missing: every timeout above this line stops
# ONE hop (a model call, an HTTP call to a service), and MAX_CRITIQUE_ITERATIONS
# stops the loop after a fixed COUNT of cycles, but nothing stops the sum of
# several hops across up to that many cycles from running for as long as each
# hop is willing to take. Under `InMemoryRunner` (adk run/adk web/Streamlit),
# ADK's own client sets no such ceiling - see genai_client.MODEL_CALL_TIMEOUT_MS.
#
# Picked from the same evidence EVALUATION.md's latency sections give, not
# invented: the 2026-08-03 20:20 final baseline (280 live runs) measured
# median latency 23.4s at critique budget 0 / 25.7s at budget 1, and its worst
# real, successfully-completed run - a genuine second critique cycle, not a
# hang - was 89.5s. 240s is a little under 2.7x that worst observed case,
# enough headroom for a legitimate third cycle (MAX_CRITIQUE_ITERATIONS allows
# one, and the Streamlit slider lets a user request it) without being sized
# for the eval harness's different purpose: run_eval.py's own
# DEFAULT_TIMEOUT_S=300s is deliberately generous enough to let even a known,
# ~100s-costing defect run to completion so the sweep still captures full
# data - a live UI should give up sooner than that and say so, not wait out a
# pathological run on the user's behalf.
TURN_TIMEOUT_S = 240.0

# Tavily, the phase 2 web search provider (src/tools/web_search.py, ADR-0028).
# The one API key in a project that otherwise authenticates everything through
# Application Default Credentials - Tavily is not a Google service, so ADC has
# nothing to offer it. Read here rather than at the call site so a blank value
# is a tool-level "not configured" error rather than an exception, and so the
# key never has to be spelled twice.
#
# Deliberately NOT validated at import time, for the same reason GCP_PROJECT
# above is not: this module is imported by tests and by code paths that never
# search the web, and failing here would break hermetic offline runs. The real
# check is in web_search.tavily_search, at the point a search is actually
# about to be made.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Bounds the whole Tavily hop, for the same reason NEWS_AGENT_TIMEOUT_S and
# MCP_FETCH_TIMEOUT_S bound theirs: it is not a Vertex call, so none of
# genai_client's model-call timeouts apply to it. This one sits in the hot
# path of every web-search answer, so it has to degrade to a structured error
# rather than stall the turn. 15s against a documented typical response of
# 1-3s for a basic search: generous enough that a slow-but-working search
# still returns, tight enough that a wedged provider costs a fraction of
# TURN_TIMEOUT_S rather than most of it.
TAVILY_SEARCH_TIMEOUT_S = 15.0

# Results requested per search. Tavily's own default is 5 and its maximum 20.
# Kept at 5 because these results are fed to a summarising sub-agent whose
# output then enters research_agent's context: more results is more input
# tokens on every web turn (MAX_SESSION_TOKENS) for diminishing marginal
# coverage, since the provider ranks by relevance.
TAVILY_MAX_RESULTS = 5
