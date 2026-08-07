"""News Agent - phase 5's single-purpose, independent agent.

Its only job: given a topic, search the public web via Google Search
grounding (the same mechanism as src/tools/web_search.py's
web_search_agent) and report the latest news on it. It exists to
demonstrate cross-process delegation (see server.py and
src/tools/news_agent.py), so unlike web_search_agent it is NOT wrapped in
an AgentTool and never sits in another agent's tools=[] list in this
process - it is spoken to directly, by this service's own Runner
(server.py), and reached from the main agent's process only over HTTP.
That also means it does not need web_search_agent's runtime-tunable
thinking budget or the session-state plumbing that supports it (see that
module's _apply_thinking_budget docstring) - there is no parent session
here to read a per-request override from, only this service's own Runner
calling it once per HTTP request, so a single fixed budget is enough.

Reuses web_search_agent's grounding-source fix (_append_grounding_sources
below): google_search grounding puts source URLs in
grounding_metadata.grounding_chunks, never in the model's own text, so
without reattaching them here the model either omits sources or invents
"(Google Search)" as one - a defect already measured once in this codebase
(see web_search.py's docstring for the full story). Trimmed to this
agent's own needs rather than imported, since the two agents' configs
(model, generate_content_config, instruction) are deliberately separate
objects - copying the one piece of logic that is genuinely shared, not the
whole module.
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import google_search
from google.genai import types

from src import config

# genai_client lives under src/services, which is otherwise this project's
# main-agent service layer, and this module runs in its own process
# (server.py, reached only over A2A/HTTP - see module docstring above). That
# is a real process boundary but not a real package boundary: this file
# already imports src.config (the line above), so the news service is
# already coupled to the shared src namespace rather than to a package of
# its own. genai_client adds nothing beyond that - it is a small module (one
# constant used below, plus a get_client() this agent never calls) with no
# side effects at import time: get_client() is lru_cache-decorated but never
# invoked at module scope, so importing this module makes no network call
# and needs no Application Default Credentials (verified by reading
# genai_client.py itself, not by inference - see
# tests/test_model_call_bounds.py, which imports this module offline and
# would fail its own setup if that stopped being true). Importing the
# constant is therefore judged appropriate: the alternative, duplicating
# 120_000 as a second literal in this file, would recreate exactly the drift
# this finding is about - two numbers that are supposed to be the same
# bound, free to go out of sync the next time one of them changes.
from src.services import genai_client

NEWS_AGENT_NAME = "news_agent"

# See web_search.py's measured comparison (thinking_budget=512 vs Gemini's
# "automatic" budget: 9.46s vs 19.95s median for the same google_search
# probe) - a low, fixed budget suits a quick news lookup and there is no
# per-request caller here to ask for anything different (see module
# docstring).
_THINKING_BUDGET = 512

_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    max_output_tokens=2048,
    # Same runaway-repetition safety net as research_agent and
    # web_search_agent (see agent.py/web_search.py) - any agent on this
    # model can in principle hit a decoding loop, so it is not assumed to
    # be specific to those two.
    frequency_penalty=0.4,
    # ADR-0013 bounded "every agent model call" at 120s via
    # genai_client.MODEL_CALL_TIMEOUT_MS, and this agent was the fourth
    # model-calling agent, added after that rule and missed by it (see
    # audit.md finding 11): research_agent, critique_agent and
    # web_search_agent all carry this same http_options, this one did not.
    # ADK builds its own genai.Client per agent and sets no timeout of its
    # own (see that constant's docstring), so without this the model call
    # backing this agent has no ceiling at all.
    #
    # This bound and the client-side one in src/tools/news_agent.py (the
    # `timeout=config.NEWS_AGENT_TIMEOUT_S` constructor field on
    # _ReachableRemoteA2aAgent, a RemoteA2aAgent subclass - not a client
    # object; exactly 20.0s) look like duplicates but cover different
    # things, in different units:
    # - MODEL_CALL_TIMEOUT_MS (here, milliseconds) bounds what THIS SERVER
    #   process spends waiting on Vertex for one model call.
    # - NEWS_AGENT_TIMEOUT_S (news_agent.py, seconds) bounds what the CALLER
    #   (research_agent's process, over A2A/HTTP) spends waiting on this
    #   service as a whole.
    # Before this line existed, a stalled Vertex call in this process had no
    # ceiling of its own: the caller's 20s timeout only made research_agent
    # give up and move on - it could not, and still cannot, reach across the
    # process boundary to cancel the abandoned model call still running here.
    # Two bounds are needed for exactly that reason: the caller's bound
    # protects the caller's turn, this one protects this service's own
    # resources, and only this one can.
    http_options=types.HttpOptions(timeout=genai_client.MODEL_CALL_TIMEOUT_MS),
    thinking_config=types.ThinkingConfig(thinking_budget=_THINKING_BUDGET),
)


def _append_grounding_sources(callback_context: CallbackContext) -> types.Content | None:
    """Append grounding source URLs to this agent's answer text.

    Same defect and same fix as web_search.py's _append_grounding_sources -
    see that function's docstring for the full measured story. Returning
    Content (not mutating in place) carries the ORIGINAL answer text
    forward with sources appended, which is what ADK honours here; a
    callback that returned only the source list would replace the answer
    with its own footnotes.
    """
    events = getattr(callback_context.session, "events", None) or []

    sources: dict[tuple[str, str], None] = {}
    answer_parts: list[str] = []
    for event in events:
        if event.author != NEWS_AGENT_NAME:
            continue
        if event.content and event.content.parts:
            answer_parts.append("".join(part.text or "" for part in event.content.parts))
        metadata = getattr(event, "grounding_metadata", None)
        for chunk in (getattr(metadata, "grounding_chunks", None) or []) if metadata else []:
            web = getattr(chunk, "web", None)
            if web and web.uri:
                sources.setdefault((web.title or "source", web.uri), None)

    answer = "".join(answer_parts).strip()
    if not sources or not answer:
        # No grounding, or no text to attach to - returning None leaves the
        # turn exactly as it was, same reasoning as web_search.py's version.
        return None

    lines = "\n".join(f"- {domain} ({uri})" for domain, uri in sources)
    return types.Content(
        role="model",
        parts=[types.Part(text=f"{answer}\n\nSources:\n{lines}")],
    )


news_agent = Agent(
    name=NEWS_AGENT_NAME,
    model=config.GEMINI_MODEL,
    description="Searches the public internet for the latest news on a given topic.",
    instruction="""You are given a topic. Use Google Search to find the
latest news about it. Report back a short list of the most recent,
relevant developments - what happened, and when, if a date is available.
State each fact once; never repeat a sentence or phrase.""",
    tools=[google_search],
    generate_content_config=_GENERATE_CONTENT_CONFIG,
    after_agent_callback=_append_grounding_sources,
)
