"""Regression check for the A2A error-recovery path in `src/tools/news_agent.py`.

Not a pytest suite, matching `scripts/verify_agent.py`'s convention: this
repo has no tests/ directory (see CLAUDE.md - the capstone's focus is agent
architecture, not RAG/eval engineering), so this is a standalone script that
prints its findings for a human to read, and exits non-zero if an assertion
fails.

## What this closes

`agent_docs/TODOS.md` had an open item: "Handle `A2AClientTimeoutError` - it
has no `status_code` and killed 1 run of 280." ADR-0022 (2026-08-06) added
`_ReachableRemoteA2aAgent` to fix the closely related bare-`A2AClientError`
defect. Its docstring originally claimed the timeout case was covered as a
side effect, by the same `except Exception` wrapper - that claim turned out
to be half right: the crash is prevented, but until this script's second
finding, the recovery text told the planner the service "must be running -
start it with `docker compose up -d news-agent`", which is a false diagnosis
for a timeout (the service accepted the connection and answered late; the
2026-08-03 sweep hit this as 1 slow run in 280, not a down service). This
script verifies both findings empirically, against the installed
`a2a`/`google-adk` packages rather than by reasoning about them:

1. Reproduces the defect on a *plain* `RemoteA2aAgent` (no wrapper): the
   installed `google/adk/agents/remote_a2a_agent.py::_run_async_impl` catches
   `_compat.A2A_HTTP_ERRORS` (== `(A2AClientError,)` in this a2a version) and
   then reads `e.status_code` unconditionally. `A2AClientTimeoutError` and the
   bare `A2AClientError` both derive from `A2AClientError` without adding a
   `status_code` attribute, so that read raises `AttributeError` from inside
   the `except` clause - which escapes the whole `try` statement rather than
   being caught by the sibling `except Exception` below it.
2. Confirms `_ReachableRemoteA2aAgent` (the one actually wired into
   `research_agent`) survives both injections and yields exactly one `Event`
   carrying real *content* (not just `error_message`), because that is the
   only field `AgentTool.run_async` reads when building the planner's tool
   result - see `news_agent.py`'s module docstring and
   `AgentTool.run_async`'s `last_content = event.content` line in the
   installed `google/adk/tools/agent_tool.py` - and that the two cases
   produce genuinely different, individually correct wording rather than the
   same generic text.
3. A deeper check against the real News Agent service (skipped, not failed,
   if it is not reachable): resolves `news_remote_agent` - the actual
   module-level singleton `research_agent` calls - against the live service
   for real, then makes the *transport* time out (`httpx.AsyncClient.send`
   raises `httpx.TimeoutException`) rather than injecting an already-built
   `A2AClientTimeoutError`. This proves the conversion chain
   (`httpx.TimeoutException` -> `A2AClientTimeoutError`, in the installed
   `a2a/client/transports/http_helpers.py`) also lands on the timeout
   wording, not just the hand-injected exception object.

Checks 1-2 inject at `google.adk.agents.remote_a2a_agent._compat.send_message`,
the exact call site inside the real `try`/`except` block, rather than a
hand-rolled stand-in for it - the point is to run the installed generator,
not a mock of it.

Run with: uv run python -m scripts.verify_news_agent_error_handling
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from google.adk.agents import remote_a2a_agent as remote_a2a_agent_module
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types as genai_types

from src import config
from src.tools.news_agent import _ReachableRemoteA2aAgent, news_remote_agent

APP_NAME = "verify-news-agent-error-handling"
USER_ID = "verify-news-agent-error-handling"
# A file path source is used instead of a real URL so nothing here depends on
# the News Agent service being reachable - `_ensure_resolved` is short
# circuited below before any card would need to be fetched.
FAKE_AGENT_CARD_SOURCE = "/nonexistent/agent-card.json"

# The "must be running" remedy is correct for a down service and wrong for a
# timeout - its presence/absence is what distinguishes the two recovery
# messages below, not just non-emptiness.
NOT_RUNNING_PHRASE = "must be running"
TIMED_OUT_PHRASE = "did not respond within"


async def _make_ctx(agent) -> InvocationContext:
    """Build a minimal but real InvocationContext with one user turn in it.

    A user-authored event with content is required so
    `_construct_message_parts_from_session` (called before the network hop)
    has something to convert into an A2A message part - an empty session
    short circuits before the code path under test is even reached.
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    await session_service.append_event(
        session,
        Event(
            author="user",
            invocation_id="verify-invocation",
            content=genai_types.Content(
                role="user", parts=[genai_types.Part(text="What is the latest news on interest rates?")]
            ),
        ),
    )
    return InvocationContext(
        session_service=session_service,
        invocation_id="verify-invocation",
        session=session,
        agent=agent,
    )


@asynccontextmanager
async def _injected_send_message(make_exc: Callable[[], Exception]):
    """Patch the exact call site `_run_async_impl` awaits, to raise a fresh exception.

    Patching `_compat.send_message` (an async generator function) rather than
    something further down the stack keeps the injection at the real
    try/except boundary documented in `_ReachableRemoteA2aAgent`'s docstring,
    so the test exercises that boundary rather than a different one.

    Takes a factory rather than a pre-built exception instance: raising the
    same exception object twice (once for the plain-agent crash check, once
    for the wrapped-agent recovery check) mutates its `__traceback__` and
    `__context__` in place, which would be actively misleading once those
    attributes are inspected below.
    """

    async def _raise(*args, **kwargs):
        raise make_exc()
        yield  # pragma: no cover - makes this an async generator function

    with patch.object(remote_a2a_agent_module._compat, "send_message", _raise):
        yield


def _resolved_agent(cls: type[RemoteA2aAgent], name: str) -> RemoteA2aAgent:
    """Construct an agent and mark it already-resolved.

    `_ensure_resolved` short circuits on `self._is_resolved and
    self._a2a_client` (remote_a2a_agent.py) before touching the network, so
    setting both directly skips card resolution entirely - this test is about
    the RPC-failure path, not the card-fetch path (which is the other,
    already-handled failure shape documented in the class docstring).
    """
    agent = cls(
        name=name.replace("-", "_"), agent_card=FAKE_AGENT_CARD_SOURCE, description="test", timeout=1.0
    )
    agent._is_resolved = True
    agent._a2a_client = object()
    return agent


async def _collect_events(agent: RemoteA2aAgent) -> list[Event]:
    ctx = await _make_ctx(agent)
    return [event async for event in agent._run_async_impl(ctx)]


async def _expect_crash(make_exc: Callable[[], Exception], label: str) -> None:
    """Reproduce the defect on a plain, unwrapped `RemoteA2aAgent`."""
    agent = _resolved_agent(RemoteA2aAgent, f"plain-{label}")
    async with _injected_send_message(make_exc):
        try:
            await _collect_events(agent)
        except AttributeError as caught:
            assert "status_code" in str(caught), (
                f"[{label}] plain RemoteA2aAgent raised AttributeError, but not the expected "
                f"'status_code' one - got: {caught!r}"
            )
            print(f"[{label}] plain RemoteA2aAgent: confirmed crash - {type(caught).__name__}: {caught}")
            return
    raise AssertionError(
        f"[{label}] plain RemoteA2aAgent did not crash - either the a2a/adk defect "
        "no longer reproduces (upgrade landed), or the injection did not reach "
        "the intended code path. Either way, re-check this script against the "
        "installed packages before trusting the wrapped-agent result below."
    )


async def _expect_readable_recovery(
    make_exc: Callable[[], Exception], label: str, *, expect_phrase: str, forbid_phrase: str
) -> None:
    """Confirm `_ReachableRemoteA2aAgent` turns the same injection into the right text.

    Not just "non-empty" - `expect_phrase` must be present (the correct
    diagnosis for this failure shape) and `forbid_phrase` must be absent (the
    other shape's diagnosis, which would be a false one here).
    """
    agent = _resolved_agent(_ReachableRemoteA2aAgent, f"reachable-{label}")
    async with _injected_send_message(make_exc):
        events = await _collect_events(agent)

    assert len(events) == 1, f"[{label}] expected exactly one recovery event, got {len(events)}: {events}"
    event = events[0]
    assert event.content is not None, f"[{label}] recovery event has no content - AgentTool.run_async would see ''"
    assert event.content.parts, f"[{label}] recovery event content has no parts"
    text = "".join(part.text or "" for part in event.content.parts)
    assert text.strip(), f"[{label}] recovery event text is empty"
    assert expect_phrase in text, f"[{label}] recovery text missing expected phrase {expect_phrase!r}: {text!r}"
    assert forbid_phrase not in text, (
        f"[{label}] recovery text contains the wrong diagnosis {forbid_phrase!r} - "
        f"this failure shape should not say that: {text!r}"
    )
    print(f"[{label}] _ReachableRemoteA2aAgent: correct recovery text -> {text!r}")


async def _check_live_service_timeout() -> None:
    """Deeper check: let a real httpx timeout convert to A2AClientTimeoutError.

    Skips (does not fail) if the News Agent service is not reachable, since
    this check needs a real resolved agent card - the other checks above
    already cover the fully offline case.
    """
    print(f"\n{'=' * 80}\nlive-service httpx.TimeoutException\n{'-' * 80}")
    try:
        await news_remote_agent._ensure_resolved()
    except Exception as exc:  # noqa: BLE001 - this check is best-effort
        print(f"SKIPPED: News Agent service not reachable for card resolution ({exc}). "
              "Start it with `docker compose up -d news-agent` to exercise this check.")
        return

    ctx = await _make_ctx(news_remote_agent)

    async def _raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("simulated transport timeout")

    with patch.object(httpx.AsyncClient, "send", _raise_timeout):
        events = [event async for event in news_remote_agent._run_async_impl(ctx)]

    assert len(events) == 1, f"expected exactly one recovery event, got {len(events)}: {events}"
    text = "".join(part.text or "" for part in events[0].content.parts)
    assert TIMED_OUT_PHRASE in text, f"missing timeout phrase: {text!r}"
    assert NOT_RUNNING_PHRASE not in text, f"wrongly diagnosed a live timeout as a down service: {text!r}"
    print(f"real transport timeout -> correct recovery text -> {text!r}")


async def main() -> None:
    cases = [
        (
            lambda: A2AClientTimeoutError("Client Request timed out"),
            "A2AClientTimeoutError",
            TIMED_OUT_PHRASE,
            NOT_RUNNING_PHRASE,
        ),
        (
            lambda: A2AClientError("Network communication error: connection refused"),
            "A2AClientError",
            NOT_RUNNING_PHRASE,
            TIMED_OUT_PHRASE,
        ),
    ]
    for make_exc, label, expect_phrase, forbid_phrase in cases:
        print(f"\n{'=' * 80}\n{label}\n{'-' * 80}")
        await _expect_crash(make_exc, label)
        await _expect_readable_recovery(make_exc, label, expect_phrase=expect_phrase, forbid_phrase=forbid_phrase)

    await _check_live_service_timeout()

    print(f"\n{'=' * 80}\nAll checks passed: both exception types crash the plain RemoteA2aAgent "
          "and are recovered into distinct, correct text by _ReachableRemoteA2aAgent.")


if __name__ == "__main__":
    asyncio.run(main())
