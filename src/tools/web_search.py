"""Web Search Tool - the phase 2 key design task.

Wraps ADK's built-in `google_search` grounding tool (Vertex AI) rather than a
custom function tool against a third-party search API. Reasoning: ADK does
not allow built-in tools to be combined with plain function tools on the same
agent, so `google_search` has to live on its own small agent; that sub-agent
is then exposed to the root agent as a callable tool via `AgentTool`, sitting
alongside `search_documents` in the root agent's tool list. This needed no
new API key/config (Vertex AI is already wired up for phase 1) at the cost of
this one extra indirection layer - the trade-off was chosen for speed and to
avoid provisioning a third-party search API key under time pressure; revisit
if we need more control over the search provider or result format later.
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from src import config


def _append_grounding_sources(callback_context: CallbackContext) -> types.Content | None:
    """Append the grounding sources to this sub-agent's answer text.

    This exists because of a measured, non-obvious defect: the agent was
    instructed to attribute every fact to its source URL and never did, in
    any trace, ever. The instruction was not being ignored - Gemini's
    `google_search` grounding does not put URLs in the model's TEXT at all.
    They arrive as structured `grounding_metadata.grounding_chunks`, and
    `AgentTool` passes only the sub-agent's text up to the caller, so the
    research agent never received a URL it could have cited. That is also
    where "(Google Search)" came from: told to cite a source and handed none,
    the model named the tool.

    So attribution has to be repaired here, at the boundary where the URLs
    still exist, rather than by asking either agent more firmly.

    Both the domain and the URL are emitted ("worldbank.org (https://...)").
    The raw grounding URI is an opaque vertexaisearch redirect that tells a
    reader nothing, and the domain alone is not a URL and so cannot satisfy a
    strict attribution check - only the pair is both honest to a human and
    machine-checkable.

    Returning Content rather than mutating in place because that is what ADK
    honours here, and it carries the ORIGINAL answer text forward with the
    sources appended: a callback that returned only the source list would
    replace the answer with its own footnotes.
    """
    events = getattr(callback_context.session, "events", None) or []

    # Dedupe on (domain, uri) while preserving order. Grounding commonly cites
    # the same domain for several chunks, and a repeated source list would
    # feed the very repetition the generation config exists to bound.
    sources: dict[tuple[str, str], None] = {}
    answer_parts: list[str] = []
    for event in events:
        if event.author != _WEB_SEARCH_AGENT_NAME:
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
        # No grounding (the model answered from its own knowledge) or no text
        # to attach to. Returning None leaves the turn exactly as it was -
        # inventing a source here would be worse than having none.
        return None

    lines = "\n".join(f"- {domain} ({uri})" for domain, uri in sources)
    return types.Content(
        role="model",
        parts=[types.Part(text=f"{answer}\n\nSources:\n{lines}")],
    )

# Same runaway-repetition safety net as root_agent (src/research_agent/agent.py)
# - a Gemini decoding loop can in principle hit any agent using this model, so
# it's applied here too rather than assumed to be root_agent-specific.
_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    max_output_tokens=4096,
    frequency_penalty=0.4,
)

_WEB_SEARCH_AGENT_NAME = "web_search_agent"

_web_search_agent = Agent(
    name=_WEB_SEARCH_AGENT_NAME,
    model=config.GEMINI_MODEL,
    description="Searches the public internet for information via Google Search.",
    # No longer asks the model to attribute facts to URLs: it cannot, since
    # google_search grounding never puts them in its text (see
    # _append_grounding_sources). Asking for something the model has no way to
    # supply is what produced the invented "(Google Search)" citation, so the
    # instruction now asks only for the facts and the callback supplies the
    # sources from grounding_metadata.
    instruction="""Answer the given query using Google Search. Report back the
relevant facts you find. State each fact once; never repeat a sentence or
phrase.""",
    tools=[google_search],
    generate_content_config=_GENERATE_CONTENT_CONFIG,
    after_agent_callback=_append_grounding_sources,
)

web_search_tool = AgentTool(agent=_web_search_agent)
