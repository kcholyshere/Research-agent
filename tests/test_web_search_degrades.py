"""The Tavily web search tool's failure and citation contracts (ADR-0028).

Two contracts are pinned here, both of them things the phase 2 provider swap
could break silently rather than loudly.

**Every failure is a structured error, never an exception.** `search_documents`
and `get_financial_data` both degrade to an error dict so a turn that loses one
source says so and carries on; a raising web search would instead kill the turn
from inside an `AgentTool` sub-agent, where the traceback is furthest from the
user. A missing key is deliberately its own diagnosis rather than folded in
with a failed search: "no key configured" and "the internet had nothing" are
opposite conclusions for the planner, and only one of them is a gap worth
reporting.

**Both timeout legs are bound.** `audit.md` finding 7 was exactly this defect
one tool over - `get_financial_data` bound the connect leg and left the read
leg on a library default of 300s, so a server that accepted the connection and
then wedged hung for five minutes against a config comment claiming 30. A test
that only checked "there is a timeout" would have passed against that defect,
so this asserts the *value on each leg*.

**The Sources block is emitted from the tool's response, not by the model.**
`src/research_agent/agent.py`'s INSTRUCTION tells the planner that a web result
"ends with a 'Sources:' list of domains and URLs" and instructs it to cite from
there, and `src/evaluation/metrics.py` scores citations against that shape. It
is produced deterministically by `_append_search_sources` precisely so it
cannot be forgotten or invented - the "(Google Search)" defect that callback
was originally written for. So the shape is asserted, not assumed.

Entirely offline: `httpx.AsyncClient` is replaced with a fake, so no API key,
no network hop and no Tavily quota are needed - hence no `integration` marker.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest
from google.genai import types

from src import config
from src.tools import web_search


class _FakeResponse:
    """Just enough of `httpx.Response` for the tool's own status/JSON handling."""

    def __init__(self, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not valid JSON")
        return self._payload


class _FakeClient:
    """Replaces `httpx.AsyncClient`, recording how it was constructed.

    `post` either returns a canned response or raises a canned exception, which
    is how each degrade path below is injected without a server to misbehave.
    """

    last_timeout: httpx.Timeout | None = None
    last_headers: dict[str, str] | None = None
    last_json: dict[str, Any] | None = None

    def __init__(self, response: Any = None, raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises

    def __call__(self, *, timeout: httpx.Timeout) -> "_FakeClient":
        type(self).last_timeout = timeout
        return self

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> Any:
        type(self).last_headers = headers
        type(self).last_json = json
        if self._raises is not None:
            raise self._raises
        return self._response


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """A configured key, so the missing-key branch is not what a test hits by accident."""
    monkeypatch.setattr(config, "TAVILY_API_KEY", "tvly-test-key")
    return "tvly-test-key"


async def _search(client: _FakeClient) -> dict:
    with patch.object(httpx, "AsyncClient", client):
        return await web_search.tavily_search("anything")


@pytest.mark.asyncio
async def test_missing_key_is_its_own_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset key must not look like a search that found nothing."""
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")

    # No client fake at all: reaching httpx with no key would be the defect,
    # and would fail here with a real connection attempt rather than pass.
    result = await web_search.tavily_search("anything")

    assert "error" in result
    assert "TAVILY_API_KEY" in result["error"]
    assert "not evidence that nothing was found" in result["error"]


@pytest.mark.asyncio
async def test_timeout_binds_every_leg(api_key: str) -> None:
    """Finding 7's shape, one tool over: assert the value on each leg, not just that one exists.

    `httpx.Timeout(x)` is documented to apply `x` to connect, read, write and
    pool alike, so a wedged provider costs the same bounded time as an
    unreachable one. Asserting all four is what distinguishes the fix from a
    version that bound only the connect leg.
    """
    client = _FakeClient(raises=httpx.ReadTimeout("wedged"))

    result = await _search(client)

    assert "error" in result
    assert "timed out" in result["error"]
    timeout = _FakeClient.last_timeout
    assert timeout is not None
    for leg in ("connect", "read", "write", "pool"):
        assert getattr(timeout, leg) == config.TAVILY_SEARCH_TIMEOUT_S, leg


@pytest.mark.asyncio
async def test_transport_error_degrades(api_key: str) -> None:
    client = _FakeClient(raises=httpx.ConnectError("refused"))

    result = await _search(client)

    assert "error" in result
    assert "search provider" in result["error"]


@pytest.mark.asyncio
async def test_http_error_reports_its_status(api_key: str) -> None:
    """A 401 (a revoked or mistyped key) has to be distinguishable from an empty result set."""
    client = _FakeClient(_FakeResponse(status_code=401, text="Unauthorized: invalid API key"))

    result = await _search(client)

    assert "error" in result
    assert "401" in result["error"]


@pytest.mark.asyncio
async def test_unparseable_body_degrades(api_key: str) -> None:
    """A proxy or error page in place of JSON must not raise out of the tool."""
    client = _FakeClient(_FakeResponse(payload=None, text="<html>502</html>"))

    result = await _search(client)

    assert result == {"error": "Web search returned a response that was not valid JSON."}


@pytest.mark.asyncio
async def test_empty_results_are_an_error_not_an_empty_list(api_key: str) -> None:
    """The sub-agent must be told the search found nothing, not handed a silent [] to summarise."""
    client = _FakeClient(_FakeResponse(payload={"results": []}))

    result = await _search(client)

    assert "error" in result
    assert "no results" in result["error"]


@pytest.mark.asyncio
async def test_successful_search_sends_the_key_and_keeps_three_fields(api_key: str) -> None:
    """The happy path: bearer auth, the configured result count, and no unread fields forwarded."""
    client = _FakeClient(
        _FakeResponse(
            payload={
                "results": [
                    {
                        "title": "IFC",
                        "url": "https://www.ifc.org/en/what-we-do",
                        "content": "IFC is the largest global development institution...",
                        "score": 0.98,
                        "raw_content": "a whole page of markup nothing downstream reads",
                    },
                    {"title": "no url", "content": "dropped - a citation needs a link"},
                ]
            }
        )
    )

    result = await _search(client)

    assert _FakeClient.last_headers == {"Authorization": f"Bearer {api_key}"}
    assert _FakeClient.last_json["max_results"] == config.TAVILY_MAX_RESULTS
    assert result["results"] == [
        {
            "title": "IFC",
            "url": "https://www.ifc.org/en/what-we-do",
            "content": "IFC is the largest global development institution...",
        }
    ]


def _event(author: str, parts: list[types.Part]) -> Any:
    """A minimal stand-in for an ADK `Event` - the callback reads only these three attributes."""

    class _Event:
        def __init__(self) -> None:
            self.author = author
            self.content = types.Content(role="model", parts=parts)

    return _Event()


class _FakeCallbackContext:
    def __init__(self, events: list[Any]) -> None:
        class _Session:
            pass

        self.session = _Session()
        self.session.events = events


def test_sources_block_is_built_from_the_tool_response() -> None:
    """The citation contract the INSTRUCTION and metrics.py both depend on.

    Dedupes on (domain, url) and strips `www.`, so one publisher cited by two
    results is one line, not two - a repeated source list would feed exactly
    the repetition `frequency_penalty` is set to bound.
    """
    results = [
        {"title": "A", "url": "https://www.worldbank.org/leadership"},
        {"title": "B", "url": "https://en.wikipedia.org/wiki/World_Bank_Group"},
        {"title": "C", "url": "https://www.worldbank.org/leadership"},  # duplicate
    ]
    events = [
        _event(
            web_search._WEB_SEARCH_AGENT_NAME,
            [
                types.Part(
                    function_response=types.FunctionResponse(
                        name="tavily_search", response={"results": results}
                    )
                )
            ],
        ),
        _event(web_search._WEB_SEARCH_AGENT_NAME, [types.Part(text="Ajay Banga is the president.")]),
    ]

    content = web_search._append_search_sources(_FakeCallbackContext(events))

    assert content is not None
    assert content.parts[0].text == (
        "Ajay Banga is the president.\n\n"
        "Sources:\n"
        "- worldbank.org (https://www.worldbank.org/leadership)\n"
        "- en.wikipedia.org (https://en.wikipedia.org/wiki/World_Bank_Group)"
    )


def test_a_failed_search_appends_no_sources() -> None:
    """No fabricated attribution when the tool errored - the answer is left exactly as it was."""
    events = [
        _event(
            web_search._WEB_SEARCH_AGENT_NAME,
            [
                types.Part(
                    function_response=types.FunctionResponse(
                        name="tavily_search", response={"error": "Web search timed out after 15.0s."}
                    )
                )
            ],
        ),
        _event(web_search._WEB_SEARCH_AGENT_NAME, [types.Part(text="The web search failed.")]),
    ]

    assert web_search._append_search_sources(_FakeCallbackContext(events)) is None
