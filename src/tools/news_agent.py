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

## What was given up: the clean unreachable-service error

The function tool caught `httpx.ConnectError` and returned
`{"error": "News Agent service is not running..."}`, which the planner could
read and report honestly. RemoteA2aAgent owns its transport, so a service that
is down now surfaces as an ADK-level failure rather than as a structured tool
response. That is a real regression in one narrow respect and the accepted cost
of addressing the remote as an agent - the instruction in agent.py still tells
the model to report a news-delegation failure rather than substituting a web
search, and the demo prerequisite (start the service first) is documented in
both server.py and the README.
"""

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool

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
news_remote_agent = RemoteA2aAgent(
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
