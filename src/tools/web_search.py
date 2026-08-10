"""Web Search Tool - the phase 2 key design task.

Searches the public internet through **Tavily**, a named third-party search
API, called directly over HTTPS by `tavily_search` below (ADR-0028). Until
2026-08-10 this instead wrapped Gemini's built-in `google_search` grounding,
which made real searches but was a substitution of the provider the brief's
tool list names; the swap is a provider change, and everything around it -
the tool's name, its place in the declared-plan gate, the deterministic
"Sources:" block it returns - is deliberately unchanged.

## Why this is still a sub-agent behind an AgentTool

The original reason was a constraint that no longer applies: ADK does not let
a built-in tool like `google_search` share an agent with plain function tools,
so grounding had to live on its own agent, exposed to the root agent through
`AgentTool`. `tavily_search` is an ordinary function and could sit directly in
`research_agent.tools`. The indirection is kept anyway, for reasons that
outlive the grounding:

- **It bounds what enters `research_agent`'s context.** A raw Tavily response
  is five results of title/URL/snippet prose. Flat on the root agent, that
  whole blob would land in the transcript and then be re-sent on every
  subsequent turn (see `src/research_agent/history_trim.py`). The sub-agent
  reads it once, in its own throwaway session, and passes up only the facts
  plus the source list - which is what `MAX_SESSION_TOKENS` and
  `MAX_HISTORY_TURNS` are both there to protect.
- **It owns the per-request thinking budget** (`_apply_thinking_budget`) and
  its own token accounting, neither of which has anywhere to live on a plain
  function.
- **The tool's name is load-bearing.** `AgentTool` names itself after the
  agent it wraps, so this stays `web_search_agent` - the exact string used by
  `schema.TOOL_TO_ROUTE`, `declare_plan.EVIDENCE_TOOL_NAMES`, the root
  agent's INSTRUCTION, the Streamlit UI's labels, and the evaluation's stored
  replay fixtures.

The cost of keeping it: a web turn now pays two model calls inside this
sub-agent (one to issue the search, one to summarise the results) where
grounding paid one. See ADR-0028's consequences.
"""

from urllib.parse import urlparse

import httpx
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src import config
from src.research_agent import token_budget
from src.services import genai_client
from src.tools.fact_tag import FactTaggedAgentTool

# Tavily's documented search endpoint. A module constant rather than a config
# knob because, unlike MCP_FETCH_URL/NEWS_AGENT_URL, this addresses a public
# third-party service and there is no deployment in which it differs - the
# same reason financial_data.py keeps its three Yahoo Finance URLs locally.
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# The tool name the sub-agent calls and `_append_search_sources` matches
# function-response parts on. Named once, here, because a mismatch between
# the two would silently drop every citation rather than fail.
_SEARCH_TOOL_NAME = "tavily_search"


async def tavily_search(query: str) -> dict:
    """Search the public internet for a query and return the top results.

    Args:
        query: What to search for, as a natural-language question or phrase.

    Returns:
        The ranked results, each with its title, URL, and a snippet of the
        page's content, or an error message if the search could not be made.
    """
    # A blank key is a configuration problem, not a search failure, and it is
    # worth saying so distinctly: the sub-agent can then report "web search is
    # not configured" instead of the planner concluding the internet had
    # nothing on the topic. Same reasoning as document_search.py's three
    # distinct diagnostics for a missing index versus an embedding failure.
    if not config.TAVILY_API_KEY:
        return {
            "error": (
                "Web search is not configured: TAVILY_API_KEY is unset. No search "
                "was made, so this is not evidence that nothing was found."
            )
        }

    payload = {
        "query": query,
        "max_results": config.TAVILY_MAX_RESULTS,
        # "basic" costs 1 API credit against the free tier's 1,000/month;
        # "advanced" costs 2 and mainly buys longer extracted content, which
        # this sub-agent then summarises away anyway.
        "search_depth": "basic",
    }

    # Every failure below becomes a structured error dict rather than an
    # exception, matching get_financial_data and search_documents: a turn that
    # loses one source should say so and carry on, not die. A single
    # httpx.Timeout value binds all four legs - connect, read, write, pool -
    # so a provider that accepts the connection and then wedges is bounded by
    # the same number as one that never answers at all. financial_data.py's
    # 300s hang (audit finding 7) was exactly the case where only one leg had
    # been bound, which is why this is spelled out rather than assumed.
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.TAVILY_SEARCH_TIMEOUT_S)
        ) as client:
            response = await client.post(
                _TAVILY_SEARCH_URL,
                json=payload,
                headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
            )
    except httpx.TimeoutException:
        return {
            "error": (
                f"Web search timed out after {config.TAVILY_SEARCH_TIMEOUT_S}s. "
                "No results were retrieved."
            )
        }
    except httpx.HTTPError as exc:
        return {"error": f"Web search could not reach the search provider: {exc}."}

    if response.status_code >= 400:
        # The body is truncated for the same reason financial_data.py truncates
        # its MCP error text: an upstream error page can be arbitrarily long,
        # and it is going into a model's context.
        return {
            "error": (
                f"Web search failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        }

    try:
        body = response.json()
    except ValueError:
        return {"error": "Web search returned a response that was not valid JSON."}

    # Only the three fields the sub-agent and the citation callback actually
    # use are kept. The full response also carries per-result scores, ids,
    # favicons and optional raw page content; forwarding those would put
    # tokens into the sub-agent's prompt that nothing downstream reads.
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in body.get("results", [])
        if item.get("url")
    ]
    if not results:
        return {"error": f"Web search for {query!r} returned no results."}
    return {"results": results}


def _domain_of(url: str) -> str:
    """The bare domain of a URL, for the human-readable half of a citation.

    `www.` is stripped because it is noise in a citation and would make the
    same publisher look like two sources in the dedupe below. A URL that will
    not parse falls back to itself rather than raising - a citation with an
    ugly label still points somewhere checkable.
    """
    host = urlparse(url).netloc
    return host[4:] if host.startswith("www.") else (host or url)


def _append_search_sources(callback_context: CallbackContext) -> types.Content | None:
    """Append the search sources to this sub-agent's answer text.

    This exists because of a measured, non-obvious defect that predates the
    Tavily swap: the agent was instructed to attribute every fact to its
    source URL and never did, in any trace, ever. Under `google_search`
    grounding the instruction was not being ignored - Gemini simply never puts
    source URLs in the model's TEXT, only in structured
    `grounding_metadata`, and `AgentTool` passes just the text up to the
    caller. That is also where "(Google Search)" came from: told to cite a
    source and handed none, the model named the tool.

    Tavily does hand the model real URLs in the tool result, so it *could* now
    cite them itself. Attribution is still repaired here rather than asked
    for, deliberately: a model that is asked to reproduce URLs sometimes
    reproduces them wrongly, and an invented or mangled citation is worse than
    the tool-name citation this callback was written to eliminate. Emitting
    the source list from the tool's own response is exact by construction.

    Both halves are emitted ("worldbank.org (https://...)"): the domain alone
    is not a URL and so cannot satisfy a strict attribution check, and the URL
    alone reads badly. This is the shape `src/research_agent/agent.py`'s
    INSTRUCTION promises the planner ("its result ends with a 'Sources:' list
    of domains and URLs") and that `src/evaluation/metrics.py` was tuned
    against, so it is kept exactly.

    The redirect-resolution machinery this callback used to carry is gone with
    the grounding that needed it: Tavily returns real destination URLs, so
    there is nothing opaque left to resolve.

    Reading the sub-agent's own function-response events is safe here because
    `AgentTool.run_async` creates a FRESH in-memory session per call (verified
    by reading the installed google-adk 2.5.0's `agent_tool.py`, not docs), so
    these events belong to this search and no earlier one.

    Returning Content rather than mutating in place because that is what ADK
    honours here, and it carries the ORIGINAL answer text forward with the
    sources appended: a callback that returned only the source list would
    replace the answer with its own footnotes.
    """
    events = getattr(callback_context.session, "events", None) or []

    # Dedupe on (domain, url) while preserving order. Several results commonly
    # share a publisher, and a repeated source list would feed the very
    # repetition the generation config exists to bound.
    sources: dict[tuple[str, str], None] = {}
    answer_parts: list[str] = []
    for event in events:
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            response = getattr(part, "function_response", None)
            if response is not None and response.name == _SEARCH_TOOL_NAME:
                for item in (response.response or {}).get("results", []):
                    url = item.get("url")
                    if url:
                        sources.setdefault((_domain_of(url), url), None)
        if event.author == _WEB_SEARCH_AGENT_NAME:
            answer_parts.append("".join(part.text or "" for part in event.content.parts))

    answer = "".join(answer_parts).strip()
    if not sources or not answer:
        # The search failed or returned nothing, or there is no text to attach
        # to. Returning None leaves the turn exactly as it was - inventing a
        # source here would be worse than having none.
        return None

    lines = "\n".join(f"- {domain} ({url})" for domain, url in sources)
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
# measured 9.46s for the same probe - a web lookup needs little deliberation,
# so most of that thinking time was overhead, not reasoning that changed the
# answer. Those numbers were measured on the google_search grounding path
# this module has since replaced (ADR-0028), so treat them as the reason the
# budget is pinned low rather than as current latency figures. The value here
# is the static default/fallback; _apply_thinking_budget below lets a single
# request override it.
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

    Note that this now applies to BOTH of the sub-agent's model calls (the one
    that issues the search and the one that summarises its results), where
    under grounding there was only one - see this module's docstring.
    """
    budget = callback_context.state.get(
        _THINKING_BUDGET_STATE_KEY, config.DEFAULT_WEB_SEARCH_THINKING_BUDGET
    )
    llm_request.config.thinking_config = types.ThinkingConfig(thinking_budget=budget)
    return None


_web_search_agent = Agent(
    name=_WEB_SEARCH_AGENT_NAME,
    model=config.GEMINI_MODEL,
    description="Searches the public internet for information via the Tavily Search API.",
    # Does not ask the model to attribute facts to URLs even though the Tavily
    # results now contain them: _append_search_sources emits the source list
    # from the tool's own response, which is exact, where a model reproducing
    # URLs by hand is not. See that callback's docstring.
    instruction="""Answer the given query using the tavily_search tool. Search
once; only search again with a different phrasing if the first results do not
contain what was asked for. Report back the relevant facts you find, using only
what the results say. If the tool returns an error, say plainly that the web
search failed and what it said - do not answer from your own knowledge instead.
State each fact once; never repeat a sentence or phrase.""",
    tools=[tavily_search],
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
    after_agent_callback=_append_search_sources,
)

# FactTaggedAgentTool rather than AgentTool: the declared-plan gate has to
# know which declared fact a call is serving, and nothing in
# before_tool_callback carries that (see src/tools/fact_tag.py). The subclass
# adds a required `fact` argument to the declaration and strips it again
# before the sub-agent runs, so the wrapped agent's prompt is unchanged and
# `tool.name` stays `web_search_agent`.
web_search_tool = FactTaggedAgentTool(agent=_web_search_agent)
