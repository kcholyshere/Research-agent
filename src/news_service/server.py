"""News Agent HTTP service - phase 5's Agent-to-Agent (A2A) demonstration.

Transport choice: plain HTTP (FastAPI/uvicorn), not the `a2a` protocol
library that phase 1's technical requirements name as the optional A2A
transport (https://github.com/a2aproject/A2A). Verified before choosing:
neither `a2a` nor `a2a_sdk` is installed in this venv
(`importlib.util.find_spec` returns None for both) and neither appears in
`uv.lock`, so it is not even a transitive dependency of anything else here
- it was never pulled in. Per CLAUDE.md's rule to verify any library
against what is actually installed rather than docs, adding and learning
an unverified third-party protocol library under this phase's "optional"
scope was judged worse than a small, explicit HTTP surface built on
dependencies already resolved in this venv: FastAPI, uvicorn and httpx
(the client tool's side) were already present as transitive dependencies
of google-adk itself before this phase (see uv.lock's [[package]] entry
for google-adk), so nothing new had to be resolved to run this. They are
now declared explicitly in pyproject.toml regardless, on the same
precedent as that file's pyyaml entry: a shipped-agent-path dependency
that a future adk release could silently drop should break the build, not
the A2A demo at runtime. What real A2A adds over this - an agent card for capability discovery,
standardised task/artifact semantics for a fleet of agents - is not worth
its complexity for one hardcoded endpoint with one request/response shape;
revisit if a second external agent joins and that discovery machinery
starts earning its keep.

Process boundary: this module is the News Agent's OWN process. It is never
imported by src/research_agent/agent.py or anything else running there -
the main agent's process reaches this only over HTTP, via
src/tools/news_agent.py's get_latest_news. That is the whole point of the
exercise: delegation to an independent service, not an in-process function
call dressed up as one (contrast web_search_tool, which wraps a sub-agent
living in the SAME process via ADK's AgentTool - see web_search.py).

Scope deliberately cut for a "minimal, single-purpose" service: no
Langfuse tracing (research_agent/agent.py wires it into every entrypoint
that imports that module; this process never does, so its calls are
currently untraced - acceptable for a phase 5 demo, revisit if this
service needs debugging support later), and no session reuse (see
get_news below).

Run it (from the repo root, so `src` resolves as a package):
    uv run python -m src.news_service.server
Binds 0.0.0.0:8001 by default; override with the NEWS_AGENT_PORT env var.
The main agent's client tool (src/tools/news_agent.py) points at
NEWS_AGENT_URL (src/config.py), default http://localhost:8001 - keep the
two in step if you change the port.

Demo it - with the service running in one terminal, from another:
    curl -s -X POST http://localhost:8001/news \\
        -H "Content-Type: application/json" \\
        -d '{"topic": "artificial intelligence regulation"}' | python -m json.tool
Or drive it from the main agent (a separate process/terminal - `uv run adk
web src`, `uv run adk run src/research_agent`, or the Streamlit UI): ask a
question the planner should route to the News Agent, e.g. "What's the
latest news on the EU AI Act?" (once the tool is wired into
research_agent/agent.py's tools=[] - see this phase's final report for the
exact lines).
"""

import os

from fastapi import FastAPI
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from pydantic import BaseModel

from src.news_service.agent import news_agent

APP_NAME = "news_service"
USER_ID = "a2a-client"

app = FastAPI(
    title="News Agent",
    description="Single-purpose latest-news lookup service (phase 5 A2A demo).",
)

# One Runner for the process's lifetime is safe to share across requests:
# InMemoryRunner itself holds no per-request state (mirrors src/ui/app.py's
# @st.cache_resource-cached runner for the main agent) - only the session
# created per request below does.
_runner = InMemoryRunner(agent=news_agent, app_name=APP_NAME)


class NewsRequest(BaseModel):
    topic: str


class NewsResponse(BaseModel):
    topic: str
    news: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/news", response_model=NewsResponse)
async def get_news(request: NewsRequest) -> NewsResponse:
    # A fresh session per request, not a reused one: this is a stateless
    # lookup service - an HTTP caller here has no notion of "session" to
    # begin with, and reusing one session across unrelated callers/topics
    # would let one request's conversation history leak into the model
    # context for the next, unrelated topic.
    session = await _runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=request.topic)])

    final_text = ""
    async for event in _runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        # Matched on author, same reasoning as research_agent/agent.py's UI
        # layer: this process runs a single LlmAgent today, so "the last
        # final response" and "news_agent's last final response" happen to
        # coincide - but matching on author is what stays correct if this
        # service ever grows a second sub-agent, and it costs nothing now.
        if (
            event.author == news_agent.name
            and event.is_final_response()
            and event.content
            and event.content.parts
        ):
            final_text = "".join(part.text or "" for part in event.content.parts)

    return NewsResponse(topic=request.topic, news=final_text or "No news found.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("NEWS_AGENT_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
