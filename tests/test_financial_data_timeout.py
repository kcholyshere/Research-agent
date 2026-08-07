"""The MCP fetch read-timeout gap in `src/tools/financial_data.py` (audit.md finding 7).

Verified against the installed mcp==1.28.1
(`mcp/client/streamable_http.py:686-713`): the deprecated `streamablehttp_client`
takes `timeout` (bounds the httpx client's connect/write/pool legs) and
`sse_read_timeout` (bounds a single SSE read) as separate `float | timedelta`
parameters, defaulting to 30 and 300 respectively when not passed. The call
site used to pass only `timeout=config.MCP_FETCH_TIMEOUT_S`, so
`sse_read_timeout` silently kept the library default of 300 seconds - a
`mcp-fetch` container that accepts the connection and then wedges on the
upstream Yahoo fetch would block `get_financial_data` for five minutes, not
the 30 seconds `config.MCP_FETCH_TIMEOUT_S`'s comment claims to bound.

This test asserts both keyword arguments the call site passes to
`streamablehttp_client`, not just one - a test that only checked `timeout`
would pass against the defect just as happily as against the fix, since the
defect never touched `timeout` at all. It is entirely offline: the transport
context manager and the session are both replaced with fakes so no network
hop or `mcp-fetch` container is needed, hence no `integration` marker.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest

from src import config
from src.tools import financial_data


class _FakeResult:
    """Stands in for `mcp.types.CallToolResult` - just enough to reach the
    tool's own text-joining and error-checking logic without a real server.
    """

    isError = False
    content: list[Any] = []


class _FakeClientSession:
    """Replaces `mcp.ClientSession` so the fake read/write streams from the
    patched transport never have to be understood by a real session object.
    """

    def __init__(self, read: Any, write: Any) -> None:
        self._read = read
        self._write = write

    async def __aenter__(self) -> "_FakeClientSession":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def initialize(self) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict) -> _FakeResult:
        return _FakeResult()


@pytest.mark.asyncio
async def test_get_financial_data_bounds_both_connect_and_read_timeout() -> None:
    """Regression for audit.md finding 7: both timeout kwargs must be set.

    Patches `financial_data.streamablehttp_client` (the name bound into this
    module at import time, not `mcp.client.streamable_http.streamablehttp_client`
    - patching the origin would not intercept the call the tool actually
    makes) and `financial_data.ClientSession`, so `get_financial_data` runs
    for real up to the point of building `ClientSession`, and the exact
    keyword arguments the transport is constructed with are captured.
    """
    captured_kwargs: dict[str, Any] = {}

    @asynccontextmanager
    async def _fake_streamablehttp_client(url: str, **kwargs: Any):
        captured_kwargs.update(kwargs)
        yield (object(), object(), lambda: None)

    with (
        patch.object(financial_data, "streamablehttp_client", _fake_streamablehttp_client),
        patch.object(financial_data, "ClientSession", _FakeClientSession),
    ):
        result = await financial_data.get_financial_data("stocks")

    # Sanity check the fake path was actually exercised end to end, so a typo
    # that made the patch a no-op does not read as a silent pass.
    assert "data" in result, f"fake path did not complete cleanly: {result}"

    assert "timeout" in captured_kwargs, (
        f"streamablehttp_client was not called with timeout=: {captured_kwargs}"
    )
    assert captured_kwargs["timeout"] == config.MCP_FETCH_TIMEOUT_S, (
        "connect/write/pool bound must come from config.MCP_FETCH_TIMEOUT_S, "
        f"got {captured_kwargs['timeout']!r}"
    )

    # This is the assertion that catches the actual defect: the old call site
    # never passed sse_read_timeout at all, so this key was simply absent and
    # the library default of 300s (60 * 5, streamable_http.py:690) silently
    # applied instead.
    assert "sse_read_timeout" in captured_kwargs, (
        "sse_read_timeout was not passed to streamablehttp_client - the read "
        "leg would fall back to the library default of 300s, five minutes "
        f"and 10x config.MCP_FETCH_TIMEOUT_S ({config.MCP_FETCH_TIMEOUT_S}s). "
        f"kwargs passed: {captured_kwargs}"
    )
    assert captured_kwargs["sse_read_timeout"] == config.MCP_FETCH_TIMEOUT_S, (
        "read bound must come from the same config.MCP_FETCH_TIMEOUT_S as the "
        f"connect bound, got {captured_kwargs['sse_read_timeout']!r}"
    )


def test_installed_mcp_default_sse_read_timeout_is_still_300s() -> None:
    """Pins the library default that motivated this fix.

    If a future `mcp` upgrade changes this default, this test - not just the
    regression above - is what surfaces it, since the regression above would
    keep passing regardless of what the library defaults to.
    """
    import inspect

    default = inspect.signature(financial_data.streamablehttp_client).parameters[
        "sse_read_timeout"
    ].default
    assert default == 60 * 5, (
        f"mcp's streamablehttp_client sse_read_timeout default changed to {default!r} "
        "(expected 300s / 60*5) - re-check whether this fix is still needed as written"
    )
