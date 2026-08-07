"""The A2A error-recovery path in `src/tools/news_agent.py` (ADR-0022).

Converted from `scripts/verify_news_agent_error_handling.py`, which was
written before this project had a tests directory. Nothing about the checks
changed - they were already assertion-shaped and already fully offline - only
where they live and how they are run.

## What this pins

ADR-0022 added `_ReachableRemoteA2aAgent` because the installed
`google/adk/agents/remote_a2a_agent.py::_run_async_impl` catches
`_compat.A2A_HTTP_ERRORS` (which is `(A2AClientError,)` in this a2a version)
and then reads `e.status_code` unconditionally. `A2AClientTimeoutError` and
the bare `A2AClientError` both derive from `A2AClientError` without adding a
`status_code` attribute, so that read raises `AttributeError` from inside the
`except` clause, which escapes the whole `try` statement rather than being
caught by the sibling `except Exception` below it. One unreachable News Agent
therefore killed the entire turn.

The two failure shapes need genuinely different wording, not merely non-empty
wording. "The service must be running" is the correct remedy for a refused
connection and a false diagnosis for a timeout - the 2026-08-03 sweep hit the
timeout as 1 slow run in 280, against a service that was up the whole time.
So each test asserts the right phrase is present *and* the other shape's
phrase is absent.

Injection happens at `remote_a2a_agent._compat.send_message`, the exact call
site inside the real try/except block, rather than at a hand-rolled stand-in
for it: the point is to run the installed generator, not a mock of it. That
is also why the plain-agent crash is asserted rather than assumed - if an ADK
upgrade fixes the defect upstream, this test fails and tells us the wrapper
has become redundant, instead of silently protecting against nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from google.adk.agents import remote_a2a_agent as remote_a2a_agent_module
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types

from src.tools.news_agent import _ReachableRemoteA2aAgent, news_remote_agent

APP_NAME = "test-news-agent-recovery"
USER_ID = "test-news-agent-recovery"

# A file path source is used instead of a real URL so nothing here depends on
# the News Agent service being reachable - `_ensure_resolved` is short
# circuited below before any card would need to be fetched.
FAKE_AGENT_CARD_SOURCE = "/nonexistent/agent-card.json"

NOT_RUNNING_PHRASE = "must be running"
TIMED_OUT_PHRASE = "did not respond within"


async def _make_ctx(agent: RemoteA2aAgent) -> InvocationContext:
    """A minimal but real InvocationContext with one user turn in it.

    Deliberately not the shared `tool_context` fixture: this path needs a
    user-authored event with content, because
    `_construct_message_parts_from_session` runs before the network hop and an
    empty session short circuits before the code under test is reached.
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    await session_service.append_event(
        session,
        Event(
            author="user",
            invocation_id="test-invocation",
            content=genai_types.Content(
                role="user",
                parts=[genai_types.Part(text="What is the latest news on interest rates?")],
            ),
        ),
    )
    return InvocationContext(
        session_service=session_service,
        invocation_id="test-invocation",
        session=session,
        agent=agent,
    )


@asynccontextmanager
async def _injected_send_message(make_exc: Callable[[], Exception]):
    """Patch the exact call site `_run_async_impl` awaits, to raise a fresh exception.

    Takes a factory rather than a pre-built exception instance: raising the
    same exception object twice mutates its `__traceback__` and `__context__`
    in place, which would be actively misleading once those are inspected.
    """

    async def _raise(*args, **kwargs):
        raise make_exc()
        yield  # pragma: no cover - makes this an async generator function

    with patch.object(remote_a2a_agent_module._compat, "send_message", _raise):
        yield


def _resolved_agent(cls: type[RemoteA2aAgent], name: str) -> RemoteA2aAgent:
    """Construct an agent and mark it already-resolved.

    `_ensure_resolved` short circuits on `self._is_resolved and
    self._a2a_client` before touching the network, so setting both directly
    skips card resolution entirely - these tests are about the RPC-failure
    path, not the card-fetch path, which is the other, separately-handled
    failure shape.
    """
    agent = cls(name=name, agent_card=FAKE_AGENT_CARD_SOURCE, description="test", timeout=1.0)
    agent._is_resolved = True
    agent._a2a_client = object()
    return agent


async def _collect_events(agent: RemoteA2aAgent) -> list[Event]:
    ctx = await _make_ctx(agent)
    return [event async for event in agent._run_async_impl(ctx)]


FAILURE_SHAPES = [
    pytest.param(
        lambda: A2AClientTimeoutError("Client Request timed out"),
        TIMED_OUT_PHRASE,
        NOT_RUNNING_PHRASE,
        id="timeout",
    ),
    pytest.param(
        lambda: A2AClientError("Network communication error: connection refused"),
        NOT_RUNNING_PHRASE,
        TIMED_OUT_PHRASE,
        id="unreachable",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_exc", "expect_phrase", "forbid_phrase"), FAILURE_SHAPES)
async def test_plain_remote_agent_still_crashes(
    make_exc: Callable[[], Exception], expect_phrase: str, forbid_phrase: str
) -> None:
    """The upstream defect must still reproduce, or the wrapper is redundant."""
    agent = _resolved_agent(RemoteA2aAgent, "plain_agent")
    with pytest.raises(AttributeError, match="status_code"):
        async with _injected_send_message(make_exc):
            await _collect_events(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_exc", "expect_phrase", "forbid_phrase"), FAILURE_SHAPES)
async def test_reachable_agent_recovers_with_the_right_diagnosis(
    make_exc: Callable[[], Exception], expect_phrase: str, forbid_phrase: str
) -> None:
    agent = _resolved_agent(_ReachableRemoteA2aAgent, "reachable_agent")
    async with _injected_send_message(make_exc):
        events = await _collect_events(agent)

    assert len(events) == 1, f"expected exactly one recovery event, got {len(events)}: {events}"
    event = events[0]
    # Content specifically, not error_message: `AgentTool.run_async` reads
    # `event.content` alone when building the planner's tool result, so a
    # recovery that only sets error_message reaches the planner as "".
    assert event.content is not None, "recovery event has no content - AgentTool.run_async would see ''"
    assert event.content.parts, "recovery event content has no parts"
    text = "".join(part.text or "" for part in event.content.parts)
    assert expect_phrase in text, f"recovery text missing expected phrase {expect_phrase!r}: {text!r}"
    assert forbid_phrase not in text, (
        f"recovery text contains the wrong diagnosis {forbid_phrase!r} for this failure shape: {text!r}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_transport_timeout_lands_on_the_timeout_wording() -> None:
    """Deeper check against the real service: let httpx's own timeout convert.

    The offline tests above hand-inject an already-built
    `A2AClientTimeoutError`. This one resolves `news_remote_agent` - the
    actual module-level singleton `research_agent` calls - against the live
    service, then makes the *transport* time out, proving the conversion chain
    (`httpx.TimeoutException` to `A2AClientTimeoutError`, in the installed
    `a2a/client/transports/http_helpers.py`) also lands on the timeout
    wording rather than only the hand-built exception doing so.
    """
    try:
        await news_remote_agent._ensure_resolved()
    except Exception as exc:  # noqa: BLE001 - unreachable service is a skip, not a failure
        pytest.skip(f"News Agent not reachable ({exc}). Start it with `docker compose up -d news-agent`.")

    ctx = await _make_ctx(news_remote_agent)

    async def _raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("simulated transport timeout")

    with patch.object(httpx.AsyncClient, "send", _raise_timeout):
        events = [event async for event in news_remote_agent._run_async_impl(ctx)]

    assert len(events) == 1, f"expected exactly one recovery event, got {len(events)}: {events}"
    text = "".join(part.text or "" for part in events[0].content.parts)
    assert TIMED_OUT_PHRASE in text, f"missing timeout phrase: {text!r}"
    assert NOT_RUNNING_PHRASE not in text, f"wrongly diagnosed a live timeout as a down service: {text!r}"
