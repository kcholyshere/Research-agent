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
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
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
#
# thinking_config is this sub-agent's own field, set on this module's own
# GenerateContentConfig object - deliberately NOT on research_agent's or
# critique_agent's (see agent.py/critique.py), which are separate
# GenerateContentConfig instances entirely. A direct Vertex probe (n=6)
# measured this sub-agent at a median 19.95s/3,310 thinking tokens with no
# budget set (Gemini's "automatic" thinking); pinning thinking_budget=512
# measured 9.46s for the same probe - google_search grounding needs little
# deliberation, so most of that thinking time was overhead, not reasoning
# that changed the answer. The value here is the static default/fallback;
# _apply_thinking_budget below lets a single request override it.
_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    max_output_tokens=4096,
    frequency_penalty=0.4,
    thinking_config=types.ThinkingConfig(
        thinking_budget=config.DEFAULT_WEB_SEARCH_THINKING_BUDGET
    ),
)

_WEB_SEARCH_AGENT_NAME = "web_search_agent"

# Session-state key the Streamlit UI writes to per turn (see src/ui/app.py) -
# named distinctly from critique_agent's "critique_budget" since both keys
# can be present in the same session state at once.
_THINKING_BUDGET_STATE_KEY = "web_search_thinking_budget"


def _apply_thinking_budget(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Apply this request's thinking budget to the outgoing LlmRequest.

    Verified directly against the installed google-adk==2.5.0 (ADR-0002's
    pin; confirmed via importlib.metadata.version("google-adk") in this
    venv), by reading the actual source, not docs:
    - base_llm_flow.py's _handle_before_model_callback calls a
      before_model_callback as callback(callback_context=..., llm_request=...)
      with the LlmRequest ADK is about to send - basic.py's
      _build_basic_request has already set llm_request.config =
      agent.generate_content_config.model_copy(deep=True) by the time this
      runs, i.e. a fresh, request-private copy, not a reference to this
      module's _GENERATE_CONTENT_CONFIG. Mutating llm_request.config here is
      therefore safe under concurrent turns with different budgets - each
      gets its own copy - and cannot leak into the static default.
    - Mutating llm_request.config in place and returning None lets the
      mutated request go out; returning a non-None LlmResponse here would
      instead short-circuit the call entirely (used by critique_agent's
      before_agent_callback for its own, different, purpose), which is not
      wanted here - this callback only ever adjusts the request.
    Both points, plus the state plumbing below, were confirmed empirically
    (not just by reading source) with a spied-callback probe: two turns
    through this agent with web_search_thinking_budget=0 and =1024 in the
    parent session's state_delta produced llm_request.config.thinking_config
    of 0 and 1024 respectively, each on its own distinct config object.

    Session state, not a tool argument, is how the value gets here: the
    Streamlit UI passes state_delta={"web_search_thinking_budget": n} into
    runner.run_async on the PARENT (research_agent-facing) session, the same
    pattern critique_agent's "critique_budget" uses. AgentTool.run_async (the
    object web_search_tool wraps around _web_search_agent) copies the
    parent's tool_context.state into the sub-agent's own freshly created
    session before running it (google/adk/tools/agent_tool.py, state_dict
    passed to session_service.create_session), so this callback's
    callback_context.state sees the same key without any extra plumbing.
    """
    budget = callback_context.state.get(
        _THINKING_BUDGET_STATE_KEY, config.DEFAULT_WEB_SEARCH_THINKING_BUDGET
    )
    llm_request.config.thinking_config = types.ThinkingConfig(thinking_budget=budget)
    return None


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
    before_model_callback=_apply_thinking_budget,
    after_agent_callback=_append_grounding_sources,
)

web_search_tool = AgentTool(agent=_web_search_agent)
