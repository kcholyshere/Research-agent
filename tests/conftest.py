"""Shared fixtures for the audit regression suite.

## What this suite is, and what it deliberately is not

It is one test per finding in `agent_docs/audit.md`, written from that
finding's own failure scenario. The audit listed "no test suite" first not
because it is the largest defect but because it gates the rest: on a codebase
with no tests, a fix for the declared-plan gate cannot be shown to work
without re-running an hour-long evaluation sweep. So the suite's job is to
make each audit fix verifiable in seconds, not to reach a coverage number.

It is not an attempt to test the agent's behaviour. `scripts/verify_agent.py`
already records why that does not work here - an LLM's exact tool choice is
too non-deterministic to assert on - and that is what `src/evaluation/` and
its question set exist for. Everything in `tests/` is deterministic: pure
logic, callbacks driven directly, and injected failures.

## The offline ToolContext, and why it matters

`enforce_tool_budget`, `record_declared_plan`, the token budget, the turn
deadline and the history trim are all ADK callbacks that take a `ToolContext`
or a `CallbackContext`. Verified against the installed `google-adk==2.5.0`:
both names are aliases of the same `Context` class
(`tools/tool_context.py:27`, `agents/callback_context.py:22`), and `Context`
needs only an `InvocationContext`, which in turn declares just
`session_service`, `invocation_id` and `session` as required fields.

That means every one of those callbacks can be driven with no `Agent`, no
`Runner`, no model call, no credentials, no FAISS index and no container -
state writes round-trip through `context.state` and land in
`context.actions.state_delta` exactly as they do in production. This is the
single fact that makes the audit's gate findings testable at all, so it is
recorded here rather than left to be rediscovered.

The context fixtures are synchronous even though `create_session` is a
coroutine, because most of what they are used to test (`enforce_tool_budget`
is an ordinary `def`) is synchronous too. An async fixture would force every
such test to become async for no reason. Nothing in the built context is
bound to the loop that created it - it holds an in-memory session object and
a service backed by plain dicts - so building it under a throwaway
`asyncio.run` and using it under a different loop is safe.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.tool_context import ToolContext

APP_NAME = "research-agent-tests"
USER_ID = "test-user"


def _build_context(invocation_id: str) -> InvocationContext:
    async def _build() -> InvocationContext:
        session_service = InMemorySessionService()
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        return InvocationContext(
            session_service=session_service,
            invocation_id=invocation_id,
            session=session,
        )

    return asyncio.run(_build())


@pytest.fixture
def make_tool_context() -> Callable[[], ToolContext]:
    """Factory for independent `ToolContext`s, each with its own fresh session.

    Needed wherever a test has to distinguish "the same turn" from "a
    different turn": the tool budget, the report_gap gate and the declared
    plan are all turn-scoped session state, and several audit findings are
    specifically about state leaking across that boundary (or failing to be
    reset within it).
    """
    counter = 0

    def _make() -> ToolContext:
        nonlocal counter
        counter += 1
        return ToolContext(_build_context(f"test-invocation-{counter}"))

    return _make


@pytest.fixture
def tool_context(make_tool_context: Callable[[], ToolContext]) -> ToolContext:
    """A single empty `ToolContext`, standing in for one fresh turn."""
    return make_tool_context()
