"""The News Agent as an independent A2A server (phase 5).

Transport: the A2A protocol, via ADK's own `to_a2a`. This replaces an earlier
hand-rolled FastAPI `POST /news` endpoint - see ADR-0017, superseded. That
version was written on the finding that `a2a` was not installed and concluded
HTTP was an acceptable substitute because the process boundary is real either
way. The conclusion was wrong: phase 5 names A2A, so the protocol IS the
deliverable, and "not installed" was the task rather than a reason to
substitute. `google.adk.a2a` ships inside google-adk 2.5.0 already; only the
`a2a-sdk` dependency behind it was missing, and it is one extra
(`google-adk[a2a]`, now declared in pyproject.toml).

What `to_a2a` gives us that the FastAPI version had to invent:

- An **agent card** at `/.well-known/agent-card.json`, built automatically from
  the agent's own name, description and tools. That is what makes this
  discoverable rather than merely reachable - a client points at the card and
  learns what the agent can do, instead of a human reading our source to find
  out the path is `/news` and the body key is `topic`.
- A **task lifecycle**, so a long call is a task with state rather than one
  blocking request that either returns or times out.
- **Typed message conversion** both ways, so the client receives ADK events
  rather than a JSON blob it has to reshape.

The old endpoint's whole body - a Pydantic request model, an InMemoryRunner, and
a hand-written event scan to find the agent's final response - is gone, because
`to_a2a` builds the runner and does the conversion. Note what that removes:
picking the right final response out of a multi-agent event stream is the one
thing CLAUDE.md flags as having been got silently wrong three times on this
project, and this file no longer does it by hand.

`to_a2a` returns a Starlette app, so uvicorn serves it exactly as before.

Run with:

    uv run python -m src.news_service.server

Binds 0.0.0.0:8001 by default; override with the NEWS_AGENT_PORT env var.
"""

import os

from google.adk.a2a.utils.agent_to_a2a import to_a2a

from src.news_service.agent import news_agent

PORT = int(os.getenv("NEWS_AGENT_PORT", "8001"))

# What the agent card advertises as its own reachable address, kept separate
# from the bind address on purpose. uvicorn binds 0.0.0.0 so the service is
# reachable on every interface, but 0.0.0.0 is a bind wildcard rather than a
# dialable address - advertising it in the card would hand every client an
# unusable RPC URL. localhost is correct for the single-machine demo; set this
# env var when the service and the research agent are on different hosts.
ADVERTISED_HOST = os.getenv("NEWS_AGENT_ADVERTISED_HOST", "localhost")

# Module level so `uvicorn src.news_service.server:app` works as well as running
# this file directly - to_a2a returns a plain Starlette app either way.
app = to_a2a(news_agent, host=ADVERTISED_HOST, port=PORT, protocol="http")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
