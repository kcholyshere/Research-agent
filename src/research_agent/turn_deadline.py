"""A coarse, whole-turn wall-clock ceiling - agent_docs/TODOS.md's "bound
worst-case turn duration": nothing outside src/evaluation/run_eval.py's own
`asyncio.wait_for` bounds how long a whole turn (up to MAX_CRITIQUE_ITERATIONS
research/critique cycles) is allowed to run, so `adk run`, `adk web` and the
Streamlit UI can all block indefinitely on a turn that keeps finding a reason
to go around the loop again.

Two callbacks, wired in agent.py, do this together:

- `stamp_turn_deadline` runs once per turn, on `root_agent` (the LoopAgent)'s
  own `before_agent_callback`, alongside `critique.reset_turn_state` - the
  same placement, for the same reason: a LoopAgent's before_agent_callback
  fires exactly once per call to run_async, before its internal loop over
  sub_agents starts, which is what makes "once per turn" true without
  comparing invocation ids by hand (see critique.reset_turn_state's own
  docstring). It records session-relative state, not the wall clock, i.e. how
  many seconds this turn has left, not deadline = now + budget - see the
  module-level note on _REMAINING_KEY for why that distinction matters here.
- `enforce_turn_deadline` runs on `research_agent`'s before_agent_callback,
  which fires at the START of every cycle (the first, and any refinement
  cycle a critique pass earns). If the turn's time is already up, it sets
  the same escalate action `critique.exit_loop` sets - so the LoopAgent stops
  after this cycle without running critique_agent again - and returns a plain
  Content saying so, rather than research_agent's usual answer.

What this does and does not bound, honestly:

- It is COARSE: the check only runs between cycles, so it can refuse to START
  a second or third cycle once the turn has overrun, but it cannot interrupt
  a cycle already in progress - a single very slow cycle 1 sails through
  untouched no matter how long it takes. That gap is real and is not this
  module's to close: MODEL_CALL_TIMEOUT_MS bounds one model call, and a cycle
  can make several (research_agent's own turns, plus a nested one inside
  web_search_tool's AgentTool) - see agent.py's report for which entrypoints
  additionally get the Streamlit-only `asyncio.wait_for` that DOES cut a
  single cycle off, and which do not.
- It IS still worth having on every entrypoint that reaches agent.py,
  `adk run`/`adk web` included, because it is the only bound in this project
  that fires on the SUM of cycles rather than on any one hop, and it is a
  small, additive callback rather than a rewrite of anything that already
  works.

Why the returned Content is safe to treat as research_agent's answer, given
CLAUDE.md's rule against trusting session.state["draft_answer"]: that rule
exists because draft_answer can hold planning narration rather than a real
answer, and nothing here reads it. ADK marks a before_agent_callback's
returned Content as an ordinary event authored by the agent whose callback
returned it (verified against the installed google-adk 2.5.0,
base_agent.py's `_handle_before_agent_callback` - `Event(author=self.name,
content=before_agent_callback_content, ...)`), with no function call and no
partial flag, so `event.is_final_response()` is True for it exactly the same
way it is for a normal answer. Every existing consumer already matches on
`event.author == research_agent.name and event.is_final_response()` (the
Streamlit UI, the eval harness) - so this event is picked up by that same
rule with no special-casing, and it is honest about what happened rather than
a smuggled-in partial answer: it says plainly that the turn ran out of time
and produced nothing this cycle, never a synthesized answer built from
whatever happened to be gathered so far.
"""

import time

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from src import config

# Seconds REMAINING when the turn started, not an absolute deadline
# (time.monotonic() + budget) - deliberately, so this reads correctly even if
# a test or a future caller stamps it on a clock that has since been rewound
# or patched (see scripts/verify_turn_timeout.py, which shrinks
# config.TURN_TIMEOUT_S rather than the clock). Turned into an absolute
# instant once, at stamp time, using the same monotonic clock
# genai_client.py and the other timeouts in this project already reason in.
_DEADLINE_KEY = "turn_deadline_monotonic"


def stamp_turn_deadline(callback_context: CallbackContext) -> None:
    """Record when this turn's time budget runs out. See module docstring."""
    callback_context.state[_DEADLINE_KEY] = time.monotonic() + config.TURN_TIMEOUT_S


def enforce_turn_deadline(callback_context: CallbackContext) -> types.Content | None:
    """Refuse to start another research cycle once the turn's time is up.

    Returns None (proceed as normal) on every cycle until the deadline has
    passed, then returns Content and sets `escalate` on the one cycle that
    finds it has - the same two-part shape critique._skip_critique_llm_call
    uses to short-circuit deterministically. A missing deadline (the key
    absent from state) is treated as "not yet due" rather than "already
    due": the only way it can be missing is stamp_turn_deadline not having
    run, and refusing to serve a turn because of an initialisation gap this
    module can't diagnose would be a worse failure than the coarse bound
    this module intentionally already is.
    """
    deadline = callback_context.state.get(_DEADLINE_KEY)
    if deadline is None or time.monotonic() < deadline:
        return None

    callback_context.actions.escalate = True
    return types.Content(
        role="model",
        parts=[
            types.Part(
                text=(
                    f"This turn exceeded its {config.TURN_TIMEOUT_S:g}s time budget "
                    "and was stopped before finishing its research. No answer was "
                    "produced this turn - please try again, or split the question "
                    "into smaller parts."
                )
            )
        ],
    )
