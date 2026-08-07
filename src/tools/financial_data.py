"""Financial Data Tool - the phase 3 key design task.

Fetches live market data from a fixed set of Yahoo Finance pages through the
reference MCP `fetch` server, reached over streamable HTTP. The tool takes a
*category*, not a URL: the three sources are hardcoded here so the restriction
to predefined financial websites is enforced in code rather than left to
planner-prompt compliance - the LLM cannot steer this tool to an arbitrary
page. That is why this is a plain ADK function tool acting as its own MCP
client, rather than ADK's MCPToolset exposing the server's generic `fetch`
tool directly to the agent.

The transport changed in ADR-0020 and the reason is worth keeping here.
Originally this spawned `docker run -i --rm mcp/fetch` per call and spoke
stdio down the pipe. That works from a local checkout but makes the agent a
process that orchestrates containers, so containerising the agent itself
turned into a nested-container problem: either mount the host Docker socket
into the agent container or nest a daemon. The server now runs as its own
long-lived service (`docker/mcp-fetch-bridge.Dockerfile`, the `mcp-fetch`
compose service) and this is an ordinary network client.

What that costs: the service has to be up. Previously any machine with Docker
running could serve a financial question with no setup; now `docker compose up
mcp-fetch` is a prerequisite for the financial route, in a container and from a
local checkout alike. That is a real ergonomic regression, accepted because the
alternative put container orchestration on the agent's runtime path forever.
A single transport is also the point - a stdio fallback would mean two code
paths to keep working and two sets of failure modes to reason about.

A fresh MCP session is still opened per call. The per-call container spawn it
used to pay for (around half a second) is gone; what remains is an HTTP session
handshake against an already-running server, and the server is `--stateless`
because nothing is carried between calls.
"""

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src import config

# streamablehttp_client is @deprecated in the installed mcp==1.28.1 in favour
# of streamable_http_client, but that replacement (streamable_http.py:600-681)
# dropped the timeout, sse_read_timeout, headers and auth parameters entirely
# - StreamableHTTPTransport.__init__ now warns and ignores them if passed. The
# only way to bound its timeouts is to hand it a pre-built httpx.AsyncClient
# via http_client=, and the one helper that builds one with MCP's own
# defaults (create_mcp_http_client) lives in mcp.shared._httpx_utils - an
# underscored module, not re-exported from mcp or mcp.client. Migrating would
# mean either importing that private module or hand-rolling
# httpx.AsyncClient(follow_redirects=True, timeout=...) ourselves and
# re-deriving the MCP defaults (follow_redirects, the 30s/300s split) that the
# deprecated wrapper currently tracks for us. That is a real migration, not a
# rename, so it is deferred rather than done under this fix - the deprecated
# call still works today (mcp.client.streamable_http.streamable_http_client
# is a thin wrapper the deprecated function delegates to internally) and only
# emits a DeprecationWarning.

# The phase 3 requirement's predefined sources - the only pages this tool
# can ever fetch.
_SOURCES = {
    "stocks": "https://finance.yahoo.com/markets/stocks/most-active/",
    "crypto": "https://finance.yahoo.com/markets/crypto/all/",
    "currencies": "https://finance.yahoo.com/markets/currencies/",
}

# Enough to cover the full market table on each page while trimming the tail
# of navigation/footer noise that would otherwise bloat the LLM context.
_MAX_LENGTH = 20_000

# What "the server is not reachable" looks like. httpx covers connect/read
# failures and OSError covers the layer below it (DNS, refused sockets).
_TRANSPORT_ERRORS = (httpx.HTTPError, OSError)


def _first_leaf(exc: BaseException) -> BaseException:
    """The innermost exception, unwrapping nested ExceptionGroups."""
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


def _unreachable(url: str, exc: BaseException) -> dict:
    return {
        "error": (
            f"The MCP fetch server at {config.MCP_FETCH_URL} is not reachable "
            f"({type(_first_leaf(exc)).__name__}), so live financial data is "
            "unavailable. Start it with `docker compose up -d mcp-fetch`. "
            "Report this rather than answering the question from another source."
        ),
        "source": url,
    }


async def get_financial_data(fact: str, category: str) -> dict:
    """Fetch the latest market data for one financial category from Yahoo Finance.

    Use this - not web search - for any question about current prices or
    movements of stocks, cryptocurrencies, or currency exchange rates.

    Args:
        fact: The fact from your declared plan (declare_plan) that this call
            is gathering, copied exactly as you wrote it there. The declared
            source for that fact must be this tool, or the call is refused.
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

    # An unreachable server is reported as a structured tool response rather
    # than raised, so the planner can say the financial route is unavailable
    # instead of the turn dying on a connection error. This is the one thing
    # the equivalent phase 5 client had to give up when it moved to
    # RemoteA2aAgent (see src/tools/news_agent.py) - worth keeping where the
    # tool still owns its own transport.
    # timeout bounds the connect/write/pool legs of the httpx client (the
    # normal request/response hop); sse_read_timeout bounds how long a single
    # SSE read may block waiting on the next event. Verified against the
    # installed mcp==1.28.1 (mcp/client/streamable_http.py): the deprecated
    # streamablehttp_client defaults sse_read_timeout to 60*5=300s when it is
    # not passed explicitly, which is exactly the gap audit.md finding 7
    # flagged - the call used to pass only timeout=, so a wedged upstream
    # fetch (server accepts the connection, then hangs on Yahoo) blocked for
    # 300s while config.MCP_FETCH_TIMEOUT_S's comment claimed 30s covered the
    # whole hop. Passing both from the same constant makes that comment true.
    # What this still does NOT bound: TURN_TIMEOUT_S cannot help either way,
    # because enforce_turn_deadline only runs between tool cycles and cannot
    # interrupt a hop already in flight (turn_deadline.py) - this timeout is
    # the only thing standing between a wedged mcp-fetch and a 5-minute hang.
    try:
        async with streamablehttp_client(
            config.MCP_FETCH_URL,
            timeout=config.MCP_FETCH_TIMEOUT_S,
            sse_read_timeout=config.MCP_FETCH_TIMEOUT_S,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "fetch", {"url": url, "max_length": _MAX_LENGTH}
                )
    # streamablehttp_client runs its reader and writer in an anyio task group,
    # so a refused connection arrives wrapped in an ExceptionGroup rather than
    # raised directly - catching httpx.ConnectError alone silently misses it.
    # split() unwraps nesting for us and, just as importantly, hands back
    # anything that is NOT a transport error in `rest`, which must still
    # propagate instead of being mislabelled as "server unreachable".
    except BaseExceptionGroup as group:
        matched, rest = group.split(_TRANSPORT_ERRORS)
        if matched is None or rest is not None:
            raise
        return _unreachable(url, matched)
    except _TRANSPORT_ERRORS as exc:
        return _unreachable(url, exc)

    text = "".join(getattr(block, "text", "") for block in result.content)
    if result.isError:
        return {"error": f"MCP fetch of {url} failed: {text[:500]}", "source": url}
    return {"data": text, "source": url}
