"""News Agent Client Tool - the phase 5 key design task.

An A2A client: the tool the main research agent's planner calls to
delegate a "what's the latest news on X" question to an independent
process, the News Agent service (src/news_service/server.py), over a plain
HTTP endpoint rather than a function call within this process - see that
module's docstring for why HTTP rather than the `a2a` protocol library.

Distinct from web_search_tool (src/tools/web_search.py): that tool wraps a
sub-agent living IN this process via ADK's AgentTool, one Python object
away, with no network hop and no way for it to be "not running". This tool
crosses an actual process boundary - the News Agent has its own model
calls, its own Vertex AI usage, its own lifecycle, and can be started,
stopped, or moved to another host independently of this agent. That is the
point of the demo, and also why this tool needs everything an in-process
call does not: a URL, a timeout, and a failure mode that degrades instead
of raising or hanging.

A plain function tool, not an AgentTool: there is no sub-agent to wrap
here in THIS process - the actual agent lives on the other side of the
HTTP call - so this is the financial_data.py shape (a plain function
acting as its own client over a boundary), not the web_search.py shape (an
in-process sub-agent wrapped for the planner to call directly).
"""

import httpx

from src import config


async def get_latest_news(topic: str) -> dict:
    """Get the latest news on a topic from the independent News Agent service.

    Use this - not web_search_tool - specifically when the user asks for
    recent/latest news, headlines, or current developments on a topic. It
    delegates to a separate News Agent process dedicated to that one job.

    Args:
        topic: The subject to find the latest news about.

    Returns:
        {"topic": ..., "news": ...} with the latest news found, or
        {"error": ...} if the News Agent service could not be reached or
        did not respond in time. On an error, report it to the user rather
        than answering from your own knowledge or falling back to another
        source - the gap is the honest answer.
    """
    try:
        async with httpx.AsyncClient(timeout=config.NEWS_AGENT_TIMEOUT_S) as client:
            response = await client.post(f"{config.NEWS_AGENT_URL}/news", json={"topic": topic})
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        # The most common failure in a demo of this shape: the News Agent
        # is simply not running as a separate process. Named apart from the
        # generic branch below so the planner gets an actionable, specific
        # reason rather than "something went wrong".
        return {
            "error": (
                f"Could not reach the News Agent service at {config.NEWS_AGENT_URL}. "
                "It runs as a separate process and is likely not started - "
                "see src/news_service/server.py for how to start it."
            )
        }
    except httpx.TimeoutException:
        return {
            "error": (
                f"News Agent service at {config.NEWS_AGENT_URL} did not respond "
                f"within {config.NEWS_AGENT_TIMEOUT_S}s."
            )
        }
    except httpx.HTTPStatusError as exc:
        return {
            "error": (
                f"News Agent service returned {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            )
        }
    except httpx.HTTPError as exc:
        # Catch-all for any other httpx/network failure (DNS, malformed
        # response, etc.) - never let this tool raise into the planner's
        # turn; an error dict is always something it can act on.
        return {"error": f"News Agent service request failed: {exc}"}
