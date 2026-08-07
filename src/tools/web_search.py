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

import asyncio

import httpx
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from src import config
from src.research_agent import token_budget
from src.services import genai_client
from src.tools.fact_tag import FactTaggedAgentTool

# Bounds each redirect-resolution request (see _resolve_redirect below). Lives
# in config alongside MCP_FETCH_TIMEOUT_S/NEWS_AGENT_TIMEOUT_S because it is
# the same kind of knob: a bound on a non-Vertex HTTP hop that this agent must
# degrade around rather than hang on.
_REDIRECT_RESOLVE_TIMEOUT_S = config.REDIRECT_RESOLVE_TIMEOUT_S

# Resolved redirect -> destination URL, for the process lifetime. The same
# source is commonly cited by several grounding chunks across turns, and a
# resolution that succeeded once will not change, so paying the network hop
# again is pure waste. A FAILED resolution is deliberately NOT cached (see
# _resolve_redirect) so a transient timeout gets a fresh attempt next time
# rather than permanently degrading that source for the rest of the process.
_RESOLVED_URL_CACHE: dict[str, str] = {}


async def _resolve_redirect(client: httpx.AsyncClient, uri: str) -> str:
    """Resolve one opaque vertexaisearch grounding redirect to its real destination.

    Verified empirically against a live grounding URL (see module-level
    comment above `_REDIRECT_RESOLVE_TIMEOUT_S`): a HEAD request is honoured
    by the redirector and lands on the true destination (e.g.
    worldbank.org/.../ajay-banga) in ~0.3s with no body downloaded, so HEAD
    is the primary path. GET is a fallback for any redirect chain that
    refuses HEAD (405/501), and is streamed rather than awaited-to-body so
    the fallback never pays for a page download either - only the response
    headers (the resolved `.url`) are read before the stream is closed.

    Never raises: any transport error, timeout, or non-2xx HEAD response
    falls back to the original opaque `uri`, either directly or via the GET
    retry below. A degraded citation (still a working, if ugly, link) beats
    a dropped one, and both beat failing the turn.

    The GET retry is deliberately NOT reached on `httpx.TimeoutException`.
    That exception is a subclass of `httpx.HTTPError`, so an earlier version
    of this function caught it too broadly - a HEAD that stalled and timed
    out at `_REDIRECT_RESOLVE_TIMEOUT_S` fell through to a GET that then
    paid the same timeout a second time, doubling the worst case for a host
    that was never going to answer either verb (measured: ~6.1s against a
    real stalling host). A stall is a property of the HOST, not the verb -
    retrying a request that already timed out with a different method buys
    nothing. So a HEAD timeout now returns the raw `uri` immediately.

    The retry is kept for the other `httpx.HTTPError` cases - a non-2xx HEAD
    status (405/501, a host that rejects HEAD specifically but may well
    serve GET) and other FAST transport errors (e.g. a refused connection,
    a protocol error) - because those fail quickly rather than stalling, so
    a second attempt is worth its cost and can recover a source HEAD alone
    would have lost.

    Worst-case bound per URI, with this split: a HEAD timeout returns
    immediately at ~1 x `_REDIRECT_RESOLVE_TIMEOUT_S`. A HEAD that fails
    FAST (non-2xx or a quick transport error) and is then followed by a GET
    that itself times out is bounded at ~1 x `_REDIRECT_RESOLVE_TIMEOUT_S`
    plus the HEAD's (small) fast-failure time - not 2x. The only path that
    can still approach 2x is a HEAD that fails via a non-timeout transport
    error just before its own timeout would otherwise have fired; that
    requires the transport to actively error out late rather than merely
    stall, which is a narrower condition than the stalled-host case this
    change targets.
    """
    if uri in _RESOLVED_URL_CACHE:
        return _RESOLVED_URL_CACHE[uri]

    try:
        response = await client.head(uri, follow_redirects=True)
    except httpx.TimeoutException:
        return uri  # The host stalled - a GET would just stall the same way.
    except httpx.HTTPError:
        pass  # A fast, non-timeout failure - worth a GET retry below.
    else:
        if response.status_code < 400:
            resolved = str(response.url)
            _RESOLVED_URL_CACHE[uri] = resolved
            return resolved
        # Non-2xx HEAD (e.g. 405) - fall through to the GET retry below.

    try:
        async with client.stream("GET", uri, follow_redirects=True) as response:
            resolved = str(response.url)
    except httpx.HTTPError:
        return uri  # Both attempts failed - the raw redirect is still a valid link.

    _RESOLVED_URL_CACHE[uri] = resolved
    return resolved


async def _append_grounding_sources(callback_context: CallbackContext) -> types.Content | None:
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
    The domain alone is not a URL and so cannot satisfy a strict attribution
    check, so the pair was always required - but the URL half used to be the
    raw grounding URI, an opaque vertexaisearch redirect that told a human
    reader nothing and, per src/evaluation/metrics.py's
    `_URL_WORD_RE`/`_prose_word_count` comment, bloated answers badly (one
    redirect is a ~200-character "word"; one decline answer was roughly half
    URL by word count). This now resolves each redirect to its real
    destination (`_resolve_redirect` above) before emitting it, concurrently
    across a turn's distinct URIs via `asyncio.gather` - this callback is in
    the hot path of every web-search answer, so resolving one-by-one would
    multiply, not just add, latency.

    This callback is `async def` deliberately, to do that resolution without
    blocking the event loop: confirmed against the installed google-adk
    2.5.0 by reading base_agent.py's `_handle_after_agent_callback` (not
    docs) - it calls the callback, checks `inspect.isawaitable(...)` on the
    result, and awaits it if so, so an async `after_agent_callback` is
    natively supported, no thread pool needed.

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

    # One shared client for every distinct URI in this turn, resolved
    # concurrently rather than in a loop - see the docstring above on why
    # this callback is async and why one-by-one resolution is not
    # acceptable in this hot path.
    async with httpx.AsyncClient(timeout=_REDIRECT_RESOLVE_TIMEOUT_S) as client:
        resolved_uris = await asyncio.gather(
            *(_resolve_redirect(client, uri) for _, uri in sources)
        )

    lines = "\n".join(
        f"- {domain} ({resolved})"
        for (domain, _), resolved in zip(sources, resolved_uris)
    )
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
    # ADR-0013 bounded "every agent model call" at 120s, and this one was
    # missed: research_agent and critique_agent both carried the timeout,
    # this sub-agent did not, so its model call had no ceiling at all. ADK
    # builds its own client and sets no timeout of its own (see
    # genai_client.MODEL_CALL_TIMEOUT_MS), which is exactly why the bound has
    # to be set here rather than assumed from the environment. Found while
    # adding the turn-level deadline: a turn ceiling is only as good as the
    # hops underneath it, and an unbounded hop makes the coarse deadline the
    # only thing standing between a hung call and an endless turn.
    http_options=types.HttpOptions(timeout=genai_client.MODEL_CALL_TIMEOUT_MS),
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
    # Two before-model callbacks, run in order: the budget check first, so a
    # session that is already over its ceiling is refused without the second
    # one bothering to configure a request that will not be sent. ADK accepts
    # a list here and runs them until one returns a response (verified against
    # google-adk 2.5.0's canonical_before_model_callbacks).
    #
    # The session token ceiling reaches into this sub-agent through session
    # state, which AgentTool copies in and forwards back out. Counting it here
    # matters more than anywhere else: ADR-0014 measured this one call at a
    # median 3,310 thinking tokens, the most expensive single call in the
    # system. See src/research_agent/token_budget.py.
    before_model_callback=[
        token_budget.enforce_session_token_budget,
        _apply_thinking_budget,
    ],
    after_model_callback=token_budget.accumulate_token_usage,
    after_agent_callback=_append_grounding_sources,
)

# FactTaggedAgentTool rather than AgentTool: the declared-plan gate has to
# know which declared fact a call is serving, and nothing in
# before_tool_callback carries that (see src/tools/fact_tag.py). The subclass
# adds a required `fact` argument to the declaration and strips it again
# before the sub-agent runs, so the wrapped agent's prompt is unchanged and
# `tool.name` stays `web_search_agent`.
web_search_tool = FactTaggedAgentTool(agent=_web_search_agent)
