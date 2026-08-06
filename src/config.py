import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vertex AI auth (no API keys - relies on Application Default Credentials).
# ADK reads GOOGLE_GENAI_USE_VERTEXAI/GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION
# from the environment itself; these mirrors are for our own modules.
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

# Bounds a single grounding-redirect resolution in src/tools/web_search.py.
# Gemini's google_search returns opaque vertexaisearch redirect links rather
# than destination URLs, and the sub-agent's after_agent_callback resolves
# them so a citation is checkable by a reader. That resolution sits in the hot
# path of every web-search answer, so it must degrade to the raw link rather
# than stall the turn. Measured against live grounding redirects (2026-08-06):
# HEAD resolves in 0.3-0.4s, so 3s is generous headroom while capping what one
# stuck host can cost. Resolutions run concurrently, so a turn pays the slowest
# single source, not the sum.
REDIRECT_RESOLVE_TIMEOUT_S = 3.0
