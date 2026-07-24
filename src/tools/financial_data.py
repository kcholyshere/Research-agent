"""Financial Data Tool - the phase 3 key design task.

Fetches live market data from a fixed set of Yahoo Finance pages through the
reference MCP `fetch` server (run as a Docker container, spoken to over
stdio). The tool takes a *category*, not a URL: the three sources are
hardcoded here so the restriction to predefined financial websites is
enforced in code rather than left to planner-prompt compliance - the LLM
cannot steer this tool to an arbitrary page. That is why this is a plain
ADK function tool acting as its own MCP client, rather than ADK's MCPToolset
exposing the server's generic `fetch` tool directly to the agent.

A fresh MCP session (and Docker container) is spawned per call: measured
overhead is around half a second, which is cheap relative to the page fetch
itself, and it avoids managing a long-lived subprocess from what may be a
short-lived CLI process. Revisit with a persistent session if latency
measurements ever say otherwise.
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# The phase 3 requirement's predefined sources - the only pages this tool
# can ever fetch.
_SOURCES = {
    "stocks": "https://finance.yahoo.com/markets/stocks/most-active/",
    "crypto": "https://finance.yahoo.com/markets/crypto/all/",
    "currencies": "https://finance.yahoo.com/markets/currencies/",
}

_SERVER = StdioServerParameters(command="docker", args=["run", "-i", "--rm", "mcp/fetch"])

# Enough to cover the full market table on each page while trimming the tail
# of navigation/footer noise that would otherwise bloat the LLM context.
_MAX_LENGTH = 20_000


async def get_financial_data(category: str) -> dict:
    """Fetch the latest market data for one financial category from Yahoo Finance.

    Use this - not web search - for any question about current prices or
    movements of stocks, cryptocurrencies, or currency exchange rates.

    Args:
        category: One of "stocks" (most-active US stocks), "crypto"
            (cryptocurrencies), or "currencies" (foreign exchange rates).

    Returns:
        The page content as markdown (a table of symbols, prices, and
        changes) plus the source URL, or an error message for an unknown
        category or a failed fetch.
    """
    url = _SOURCES.get(category)
    if url is None:
        return {
            "error": f"Unknown category {category!r}. Valid categories: {sorted(_SOURCES)}."
        }

    async with stdio_client(_SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "fetch", {"url": url, "max_length": _MAX_LENGTH}
            )
            text = "".join(getattr(block, "text", "") for block in result.content)
            if result.isError:
                return {"error": f"MCP fetch of {url} failed: {text[:500]}", "source": url}
            return {"data": text, "source": url}
