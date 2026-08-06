"""The A2A client side of phase 5: delegating to an independent News Agent.

This replaces an httpx function tool that POSTed to a hand-rolled `/news`
endpoint (ADR-0017, superseded). The service now speaks the A2A protocol - see
`src/news_service/server.py` - and this is the matching client.

## Why a RemoteA2aAgent wrapped in an AgentTool

`RemoteA2aAgent` is ADK's A2A client: give it an agent card URL and it resolves
the card, manages the HTTP client, converts messages in both directions, and
carries session state across requests. Wrapping it in `AgentTool` is what makes
it appear in `research_agent`'s tool list, which is the same composition
`web_search.py` already uses for its search sub-agent.

The alternative - keeping a plain function tool that drives the `a2a` client
library by hand - was rejected even though it would have preserved a nicer
error contract (see below). The phase asks for agent-to-agent collaboration,
and a function that calls a service is not that: the distinction it is testing
is precisely whether the main agent delegates to *another agent* rather than
invoking an endpoint. Going through RemoteA2aAgent means the remote agent is
addressed as an agent, discovered through its card rather than through
hard-coded knowledge of its request shape.

## The tool name changes, and that is the load-bearing detail

The old function tool was named `get_latest_news`, because ADK takes a function
tool's name from `__name__`. `AgentTool` does not work that way: it names itself
after the agent it wraps (`super().__init__(name=agent.name)`), so the name that
appears in an event's `call.name` is `NEWS_AGENT_NAME` - "news_agent".

This is the exact trap documented in `src/evaluation/schema.py`'s TOOL_TO_ROUTE
comment, where guessing the variable name instead of the agent name silently
reported zero routes used for every web question - a naming mismatch that looked
like a genuine routing defect. So the name is exported here as
`NEWS_AGENT_TOOL_NAME`, taken from the agent object rather than written as a
literal, and both `schema.TOOL_TO_ROUTE` and `research_agent`'s instruction
refer to it by that value. Verified against a real turn, not assumed.

## Recovering the clean unreachable-service error

The function tool caught `httpx.ConnectError` and returned
`{"error": "News Agent service is not running..."}`, which the planner could
read and report honestly. RemoteA2aAgent owns its transport, so that structured
response was given up when this moved to A2A - and it turned out to be worse
than a lost niceity. `_ReachableRemoteA2aAgent` below restores it; see its
docstring for the ADK defect that made this necessary rather than merely nice.
"""

from collections.abc import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from src import config
from src.news_service.agent import NEWS_AGENT_NAME

# The agent card path is fixed by the A2A specification, and `to_a2a` serves the
# card there automatically. Pointing the client at the card rather than at an RPC
# endpoint is the whole point of discovery: the client learns the RPC URL, the
# agent's description and its capabilities from the card, so none of that has to
# be duplicated on this side.
AGENT_CARD_URL = f"{config.NEWS_AGENT_URL}/.well-known/agent-card.json"

# The card is fetched lazily, on the first call rather than at import. That
# matters more than it looks: this module is imported by research_agent.agent,
# which every entrypoint imports - so resolving the card eagerly would make
# `adk web`, the Streamlit UI and the eval harness all fail to start whenever
# the News Agent happens not to be running, instead of failing only the news
# questions. ADK's RemoteA2aAgent already defers resolution, which is why
# passing the URL as a string is the correct usage here and not a shortcut.
class _ReachableRemoteA2aAgent(RemoteA2aAgent):
    """RemoteA2aAgent that degrades when the service is unreachable.

    ## The defect

    In google-adk 2.5.0, `RemoteA2aAgent._run_async_impl` has a handler whose
    whole job is to turn a failed A2A request into a clean error event:

        except _compat.A2A_HTTP_ERRORS as e:
            ...
            A2A_METADATA_PREFIX + "status_code": str(e.status_code),

    `A2A_HTTP_ERRORS` is `(A2AClientError,)` - the base class - but only its
    subclass `A2AClientHTTPError` carries `status_code`. A transport failure
    (DNS, connection refused: exactly the "service is not running" case) raises
    the bare base class, so the handler raises `AttributeError` while handling
    the error. That escapes the `except Exception` below it, because it is
    raised from inside a sibling `except` block rather than from the `try`.

    The result is not a degraded answer, it is a dead turn: the whole
    invocation dies with an `AttributeError` about `status_code` that names
    nothing about the actual problem. Reproduced against the running stack on
    2026-08-06; verified by reading the installed package, not the docs.

    ## The second failure shape, which looks harmless and is worse

    A service that is down fails in two different places depending on how far
    the client got, and only one of them crashes:

    - **The card cannot be fetched** (the service was never up). ADK catches
      this itself and yields an event carrying `error_message` and no content.
      No crash - but `AgentTool.run_async` builds its return value from the
      last event's *content*, so this reaches the planner as an empty string.
      A tool that appears to have succeeded and found nothing is precisely the
      reading that has it substitute a web search and present the result as
      news, which is the defect ADR-0015 spent a sweep measuring.
    - **The card resolves but the RPC fails** (the address in the card is not
      reachable from here - the exact compose-vs-host case fixed on the server
      side). This is the `AttributeError` above.

    Both are handled, and both produce the same thing: an event with real text.

    ## The fix, and why it is shaped like this

    Delegate to `super()` and repair what comes back. Nothing is reimplemented
    - every successful path is ADK's own code, so an upgrade that fixes the
    defect upstream simply makes the `except` unreachable rather than
    conflicting with it.

    The recovery event carries text content rather than `error_message`, for
    the reason set out above: content is the only field the planner will
    actually see through `AgentTool`. It says what happened and what it means,
    in the same spirit as `financial_data.py`'s structured unreachable-server
    response.
    """

    def _unreachable(self, ctx: InvocationContext, detail: str) -> Event:
        return Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            "The News Agent service could not be reached at "
                            f"{config.NEWS_AGENT_URL} ({detail}). It is a separate "
                            "service and must be running - start it with "
                            "`docker compose up -d news-agent`. No news was retrieved "
                            "for this request; report that the news delegation failed "
                            "rather than answering from another source."
                        )
                    )
                ],
            ),
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        try:
            async for event in super()._run_async_impl(ctx):
                # An error event with no content would reach the planner as an
                # empty tool result - substitute one that says what went wrong.
                if event.error_message and not event.content:
                    yield self._unreachable(ctx, event.error_message)
                else:
                    yield event
        except Exception as exc:  # noqa: BLE001 - see class docstring
            yield self._unreachable(ctx, f"{type(exc).__name__}: {exc}")


news_remote_agent = _ReachableRemoteA2aAgent(
    name=NEWS_AGENT_NAME,
    agent_card=AGENT_CARD_URL,
    description=(
        "An independent News Agent, running as its own service and reached over "
        "the A2A protocol. Given a topic, it returns the latest news on it with "
        "source URLs."
    ),
    timeout=config.NEWS_AGENT_TIMEOUT_S,
)

news_agent_tool = AgentTool(agent=news_remote_agent)

# The name research_agent will actually see, and therefore the name that appears
# in an event's `call.name`. Read off the constructed tool rather than written as
# a literal, so it cannot drift from whatever AgentTool decided - which is the
# failure mode schema.py's TOOL_TO_ROUTE comment exists to prevent.
NEWS_AGENT_TOOL_NAME = news_agent_tool.name
