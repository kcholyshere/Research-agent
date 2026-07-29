"""A hard, per-turn ceiling on how many times each tool may be called.

Why this exists in code rather than in the instruction: the instruction
already forbids the behaviour this bounds. Step 2 of `agent.py`'s INSTRUCTION
says "call only the tool(s) you planned for each fact, once each", "never
issue another search to verify, confirm, or add detail beyond what was
asked", and "Reformulate and search again on the same source only if the
first results do not contain what you need". The 2026-07-29 baseline measured
`kb-charges-on-borrowings` making up to 10 `search_documents` calls for a
single figure, and 59 redundancy failures across 244 scored runs. The words
are there and they are ignored at scale.

That is the same conclusion ADR-0009 reached about repetition: config bounds
the worst case, prompts cannot. A search storm is not a judgement the model
is making badly, it is a loop with no stop condition, and a loop needs a
bound rather than a stronger request to stop.

## What the refusal says, and why that wording

Returning a value from `before_tool_callback` makes ADK skip the real tool
and hand the returned object back as the tool response (verified against the
installed google-adk 2.5.0: "When present, the returned tool response will be
used and the framework will skip calling the actual tool"). So the refusal is
not an error the model has to interpret - it is the tool's answer this time.

It deliberately does two things at once. It states the budget is spent, and
it names the correct next move: answer from what was already retrieved, or
say plainly that the fact is not there and cite the source checked. That
second half matters because the measured failure mode is not only "searches
too much" - it is "searches too much, then answers from whatever it found
last". Ending the search without naming the alternative would leave the model
to invent one, and the invented one is exactly the fall-back-to-web defect
this project is already fighting.

## The ceiling

Three calls per tool per turn. The eval's own `max_tool_calls` labels sit
between 1 and 4 across the question set, and the instruction sanctions one
reformulation on a miss, so three leaves genuine reformulation intact while
cutting a 10-call storm off at its third attempt. It is per tool rather than
per turn in total, so a multi-source question that legitimately needs the
knowledge base twice and the web once is unaffected.

Note this is a ceiling, not a target: a well-behaved turn never reaches it,
and reaching it is itself a signal worth seeing in a trace.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

# Per tool, per turn. See the module docstring for why three.
MAX_CALLS_PER_TOOL_PER_TURN = 3

# Session-state key holding {tool_name: calls_so_far} for the current turn.
# Reset by critique.reset_turn_state, which is the LoopAgent's own
# before_agent_callback and therefore the one hook in this system that fires
# exactly once per turn rather than once per refinement cycle. Counting per
# turn rather than per cycle is deliberate: a refinement cycle that re-runs
# the same three searches is precisely the waste being bounded, so the budget
# must not refill when the loop goes round again.
STATE_KEY = "tool_calls_this_turn"


def _refusal(tool_name: str, used: int) -> dict[str, Any]:
    return {
        "error": "tool_call_budget_exhausted",
        "detail": (
            f"You have already called {tool_name} {used} times this turn, which is the "
            f"limit. No further {tool_name} calls are available for this question. "
            "Answer now from what you have already retrieved. If what you retrieved "
            "does not contain the fact that was asked for, say so plainly and name the "
            "source you checked - do not substitute a different source, a different "
            "period, or a related figure."
        ),
    }


def enforce_tool_budget(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """Count this turn's calls per tool; refuse past the ceiling.

    Returning None lets the real tool run, which is the path every
    well-behaved turn takes. Returning a dict short-circuits the call.

    The counter is read, incremented and written back as a new dict rather
    than mutated in place, because ADK tracks state deltas by assignment -
    mutating the nested dict returned by `state.get(...)` would not always be
    recorded as a change.
    """
    counts = dict(tool_context.state.get(STATE_KEY) or {})
    used = counts.get(tool.name, 0)
    if used >= MAX_CALLS_PER_TOOL_PER_TURN:
        return _refusal(tool.name, used)
    counts[tool.name] = used + 1
    tool_context.state[STATE_KEY] = counts
    return None
