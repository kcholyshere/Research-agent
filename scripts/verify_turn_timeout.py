"""Manual verification for the whole-turn wall-clock ceiling (config.TURN_TIMEOUT_S,
src/research_agent/turn_deadline.py, and app.py's asyncio.wait_for wrapper).

Not a pytest suite, same reason scripts/verify_agent.py isn't one (no tests/
directory in this project - see that script's docstring). This one drives
src/ui/app.py through Streamlit's own `AppTest` harness rather than
`InMemoryRunner` directly, because the thing under test - the timeout
wrapper, the honest failure message, the "Thinking for x.xs" status and step
trace - lives IN app.py's script-level code, not in agent.py.

Two cases:

1. Forced timeout, deterministic. `InMemoryRunner.run_async` is monkeypatched
   process-wide to hang well past the timeout, and config.TURN_TIMEOUT_S is
   shrunk to a fraction of a second - the same "shrink the budget rather than
   wait out the real one" approach run_eval.py's own tests would use. This
   never calls Vertex, so it costs nothing and cannot flake on network
   latency the way a real slow turn would. Confirms: the turn is cut off
   promptly, the honest "exceeded its Ns time budget" message is what the
   chat shows (not a partial answer, not a raw exception), and the app does
   not crash.
2. A real, fast KB question, unpatched. Confirms the ordinary path - the
   status ticks "Thinking for x.xs", the step trace lists search_documents,
   and a real answer lands - still works with the wrapper in place.

Run with: uv run python -m scripts.verify_turn_timeout
"""

import asyncio
import time
from unittest import mock

from google.adk.runners import InMemoryRunner
from streamlit.testing.v1 import AppTest

from src import config

APP_PATH = "src/ui/app.py"

# Comfortably above config.TURN_TIMEOUT_S even when case 1 shrinks it to
# _FORCED_TIMEOUT_S below, and comfortably above how long a real fast KB
# question takes in case 2 - see EVALUATION.md's 2026-08-03 baselines (median
# 23-26s) for why 60s is a safe ceiling for the live case without being close
# to a real turn's typical latency.
_APPTEST_TIMEOUT_S = 60.0

# Small enough that the forced-hang case finishes in a fraction of a second
# rather than waiting out a real budget, large enough that Streamlit's own
# background-thread polling loop (app.py's 0.2s join interval) gets at least
# one full tick before the deadline fires.
_FORCED_TIMEOUT_S = 0.3


async def _hang_forever(self, *args, **kwargs):
    """Stand-in for InMemoryRunner.run_async that never yields an event.

    Must be an async GENERATOR (a bare `async def` with no `yield` returns a
    coroutine, and `async for event in runner.run_async(...)` in app.py would
    raise TypeError on that rather than hang) - the `yield` below is
    unreachable, its only job is to make this a generator function so the
    `await asyncio.sleep` before it is what actually blocks.
    """
    await asyncio.sleep(9999)
    yield  # pragma: no cover - unreachable, see docstring


def verify_forced_timeout() -> None:
    print(f"\n{'=' * 80}\n[forced-timeout] config.TURN_TIMEOUT_S patched to {_FORCED_TIMEOUT_S}s, "
          "InMemoryRunner.run_async patched to hang\n" + "-" * 80)
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
    messages = [msg.markdown[-1].value for msg in at.chat_message if msg.markdown]
    answer = messages[-1] if messages else "(no chat message found)"
    print(f"elapsed (wall clock, real time module - patched config, not patched clock): {elapsed:.2f}s")
    print(f"answer shown: {answer}")
    assert "exceeded its" in answer and "time budget" in answer, (
        "expected the honest timeout message, got something else - "
        "either the wrapper didn't fire or the message text drifted"
    )
    assert elapsed < _APPTEST_TIMEOUT_S / 2, (
        f"took {elapsed:.2f}s to cut off a {_FORCED_TIMEOUT_S}s budget - "
        "the wait_for wrapper is not actually bounding the turn"
    )
    print("PASS: cut off promptly, honest message shown, no partial answer, no crash.")


def verify_fast_question_unaffected() -> None:
    print(f"\n{'=' * 80}\n[fast-question] real turn, config.TURN_TIMEOUT_S={config.TURN_TIMEOUT_S}s "
          "(unpatched)\n" + "-" * 80)
    at = AppTest.from_file(APP_PATH, default_timeout=_APPTEST_TIMEOUT_S)
    at.run()
    started = time.monotonic()
    at.chat_input[0].set_value("What is the stated mission of the International Finance Corporation?").run()
    elapsed = time.monotonic() - started

    assert not at.exception, f"AppTest raised: {at.exception}"
    status_labels = [s.label for s in at.status]
    messages = [msg.markdown[-1].value for msg in at.chat_message if msg.markdown]
    answer = messages[-1] if messages else "(no chat message found)"
    print(f"elapsed: {elapsed:.1f}s")
    print(f"status label: {status_labels}")
    print(f"answer: {answer}")
    assert any(label.startswith("Thought for") for label in status_labels), (
        f"expected a completed 'Thought for x.xs' status, got {status_labels!r}"
    )
    assert "exceeded its" not in answer, "fast question was wrongly reported as timed out"
    print("PASS: normal turn completed, status/step trace intact, real answer shown.")


if __name__ == "__main__":
    verify_forced_timeout()
    verify_fast_question_unaffected()
    print(f"\n{'=' * 80}\nDone.")
