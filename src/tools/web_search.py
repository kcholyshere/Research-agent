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
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from src import config

# Same runaway-repetition safety net as root_agent (src/research_agent/agent.py)
# - a Gemini decoding loop can in principle hit any agent using this model, so
# it's applied here too rather than assumed to be root_agent-specific.
_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    max_output_tokens=4096,
    frequency_penalty=0.4,
)

_web_search_agent = Agent(
    name="web_search_agent",
    model=config.GEMINI_MODEL,
    description="Searches the public internet for information via Google Search.",
    instruction="""Answer the given query using Google Search. Report back the
relevant facts you find, each attributed to its source URL, so the calling
agent can cite it. State each fact once; never repeat a sentence or phrase.""",
    tools=[google_search],
    generate_content_config=_GENERATE_CONTENT_CONFIG,
)

web_search_tool = AgentTool(agent=_web_search_agent)
