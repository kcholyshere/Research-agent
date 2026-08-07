"""Checks the offline `ToolContext` fixture behaves like the real thing.

This is the one test in the suite that tests the test harness rather than the
project. It earns its place because every gate test below depends on the
claim in `conftest.py`'s docstring - that a hand-built `ToolContext` reads and
writes session state exactly as ADK's own does. If that claim ever stops
holding on an ADK upgrade, the gate tests would keep passing while asserting
against a context that no longer resembles production, which is precisely the
silent-pass shape this project has been bitten by before.
"""

from __future__ import annotations

from collections.abc import Callable

from google.adk.tools.tool_context import ToolContext


def test_state_writes_round_trip(tool_context: ToolContext) -> None:
    tool_context.state["some_key"] = {"search_documents": 1}
    assert tool_context.state.get("some_key") == {"search_documents": 1}


def test_state_writes_land_in_the_delta(tool_context: ToolContext) -> None:
    """ADK propagates callback state changes via `actions.state_delta`.

    A context whose writes never reached the delta would look correct to an
    in-test assertion and change nothing in a real session, so this is the
    part worth pinning rather than the read-back above.
    """
    tool_context.state["some_key"] = "value"
    assert tool_context.actions.state_delta["some_key"] == "value"


def test_contexts_from_the_factory_do_not_share_state(
    make_tool_context: Callable[[], ToolContext],
) -> None:
    """Two contexts must be as independent as two turns are."""
    first, second = make_tool_context(), make_tool_context()
    first.state["some_key"] = "first"
    assert second.state.get("some_key") is None
