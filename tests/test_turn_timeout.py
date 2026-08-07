"""The whole-turn wall-clock ceiling, ADR-0025 (`config.TURN_TIMEOUT_S`).

Converted from `scripts/verify_turn_timeout.py`. It drives `src/ui/app.py`
through Streamlit's own `AppTest` harness rather than `InMemoryRunner`
directly, because the thing under test - the `asyncio.wait_for` wrapper, the
honest failure message, and the fact that the app does not crash - lives in
app.py's script-level code, not in agent.py.

The forced-timeout case is deterministic and offline: `InMemoryRunner.run_async`
is patched process-wide to hang, and `TURN_TIMEOUT_S` is shrunk to a fraction
of a second - shrinking the budget rather than waiting out the real one. It
never calls Vertex, so it costs nothing and cannot flake on network latency
the way a genuinely slow turn would.

The second case is a real question against live Vertex and a live index, so it
is marked `integration`. It is what proves the wrapper does not also cut off
ordinary turns, which is the failure a timeout test cannot catch on its own.
"""

from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest
from google.adk.runners import InMemoryRunner
from streamlit.testing.v1 import AppTest

from src import config

APP_PATH = "src/ui/app.py"

# Comfortably above `TURN_TIMEOUT_S` even when the forced case shrinks it, and
# comfortably above how long a real fast KB question takes - see
# EVALUATION.md's 2026-08-03 baselines (median 23-26s) for why 60s is a safe
# ceiling for the live case without being close to a real turn's latency.
_APPTEST_TIMEOUT_S = 60.0

# Small enough that the forced-hang case finishes in a fraction of a second
# rather than waiting out a real budget, large enough that app.py's own 0.2s
# background-thread join interval gets at least one full tick before the
# deadline fires.
_FORCED_TIMEOUT_S = 0.3


async def _hang_forever(self, *args, **kwargs):
    """Stand-in for `InMemoryRunner.run_async` that never yields an event.

    Must be an async GENERATOR: a bare `async def` with no `yield` returns a
    coroutine, and app.py's `async for event in runner.run_async(...)` would
    raise TypeError on that rather than hang. The `yield` below is
    unreachable and exists only to make this a generator function, so the
    `await asyncio.sleep` above it is what actually blocks.
    """
    await asyncio.sleep(9999)
    yield  # pragma: no cover - unreachable, see docstring


def _last_answer(at: AppTest) -> str:
    messages = [msg.markdown[-1].value for msg in at.chat_message if msg.markdown]
    return messages[-1] if messages else "(no chat message found)"


def test_a_hung_turn_is_cut_off_with_an_honest_message() -> None:
    with (
        mock.patch.object(config, "TURN_TIMEOUT_S", _FORCED_TIMEOUT_S),
        mock.patch.object(InMemoryRunner, "run_async", _hang_forever),
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=_APPTEST_TIMEOUT_S)
        at.run()
        started = time.monotonic()
        at.chat_input[0].set_value("What was IFC's Net Income for fiscal year 2024?").run()
        elapsed = time.monotonic() - started

    assert not at.exception, f"AppTest raised: {at.exception}"
    answer = _last_answer(at)
    assert "exceeded its" in answer and "time budget" in answer, (
        f"expected the honest timeout message, got {answer!r} - either the wrapper did not "
        "fire or the message text drifted"
    )
    assert elapsed < _APPTEST_TIMEOUT_S / 2, (
        f"took {elapsed:.2f}s to cut off a {_FORCED_TIMEOUT_S}s budget - the wait_for wrapper "
        "is not actually bounding the turn"
    )


@pytest.mark.integration
def test_a_fast_real_question_is_unaffected() -> None:
    """The other half: a wrapper that cut off ordinary turns would also pass above."""
    at = AppTest.from_file(APP_PATH, default_timeout=_APPTEST_TIMEOUT_S)
    at.run()
    at.chat_input[0].set_value("What is the stated mission of the International Finance Corporation?").run()

    assert not at.exception, f"AppTest raised: {at.exception}"
    status_labels = [s.label for s in at.status]
    assert any(label.startswith("Thought for") for label in status_labels), (
        f"expected a completed 'Thought for x.xs' status, got {status_labels!r}"
    )
    assert "exceeded its" not in _last_answer(at), "fast question was wrongly reported as timed out"
