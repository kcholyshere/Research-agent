"""A hard ceiling on how many evidence-gathering tool calls a turn may make.

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

Five calls per turn in total, across all tools, and reaching it refuses every
tool rather than the one that ran out.

It was three per tool first, and the sweep that followed showed why that is
the wrong unit. Bounding each tool separately leaves an unspent budget on
every other tool, so a turn refused on `search_documents` simply used the web
instead: `decline-headcount-by-country` and `decline-segment-margin` went to
the web on 8 of 8 runs, total calls per turn barely moved (4.4 to 4.9), and
routing failures rose from 35 to 42. The cap bounded the storm and redirected
it. Counting the turn removes the sideways exit, which is the only thing that
made the redirection possible.

Five, because the largest legitimate need in the question set is four
(`multi-all-three` routes to all three sources with `max_tool_calls: 4`) and
the instruction sanctions one reformulation on a miss.

What this cannot do, stated plainly because the first version of this module
implied otherwise: a ceiling bounds waste, it cannot produce efficiency. A
question with `max_tool_calls: 2` still fails redundancy whenever the agent
spends four calls badly - the ceiling only stops the fifth. Making the agent
efficient rather than merely bounded needs the planner to commit to a
declared plan and the executor to be held to it, which is ADR-0015's deferred
option, not something a counter can reach.

Note this is a ceiling, not a target: a well-behaved turn never reaches it,
and reaching it is itself a signal worth seeing in a trace.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

# Total evidence-gathering calls per turn, across ALL tools.
#
# It was three PER TOOL until the 2026-07-29 post-fix sweep measured what that
# actually does. Per-tool counting bounded each tool and let the turn escape
# sideways: refused on search_documents, the agent went to the web, which had
# its own untouched budget. decline-headcount-by-country went to the web on 8
# of 8 runs, decline-segment-margin 8 of 8, and total calls per turn barely
# moved (4.4 to 4.9) while routing failures rose from 35 to 42. The cap
# bounded the storm and redirected it.
#
# So the ceiling is now the turn, and hitting it refuses EVERY tool rather
# than one. That is the whole point: the previous refusal ASKED the model not
# to substitute another source ("do not substitute a different source, a
# different period, or a related figure") and it did anyway - the fourth time
# in this project that prompt wording lost to behaviour. Removing the option
# is a bound; asking again would have been a fourth request.
#
# Five, because the largest legitimate need in the question set is four
# (multi-all-three routes to all three sources, max_tool_calls 4), and one
# reformulation on a miss is sanctioned by the instruction. Five leaves that
# intact and bounds everything else. Note what this cannot do: a ceiling
# bounds waste, it cannot produce efficiency, so questions with
# max_tool_calls of 2 will still fail redundancy whenever the agent uses four
# calls badly. Fixing that needs a planner constrained to a declared plan -
# see ADR-0015.
MAX_TOOL_CALLS_PER_TURN = 5

# Tools that are not evidence-gathering, and are therefore neither counted
# against the ceiling nor refusable by it.
#
# create_canvas (phase 6) is terminal: it contributes no fact, nothing is
# planned after it, and it runs precisely once at the end of a turn that was
# asked for a deliverable. Counting it would be wrong twice over. It would
# consume a slot that exists to bound *searching*, and - the failure that
# actually matters - a report turn plausibly spends KB search, one
# reformulation, and a web search before it renders anything, which puts
# create_canvas at or past the ceiling. Refusing it would mean the artefact
# silently does not exist while the turn still returns a perfectly ordinary
# prose answer. That is invisible from the outside: no error, no missing
# response, just a deliverable that quietly became a paragraph.
#
# report_gap is exempt for the same underlying reason, not because it is also
# terminal - it isn't, the turn continues to the prose decline after it - but
# because it retrieves nothing and a spent search budget has no bearing on
# recording an outcome about evidence already gathered. The failure mode is
# the mirror image of create_canvas's: `decline-*` questions are exactly the
# ones most likely to have already spent their budget on a reformulation and
# a stray extra search by the time the gap is confirmed, so a budget that
# could refuse report_gap would take away the one action this project added
# specifically to stop that pattern, on precisely the turns that need it.
#
# Deliberately a name-keyed set rather than a flag on the tool object: ADK's
# auto-wrapped function tools carry no field this project controls, and
# matching on the name is what `src/evaluation/schema.py` already does for
# CONTROL_TOOLS and OUTPUT_TOOLS. The two lists must agree - if a further
# evidence tool or non-evidence tool is added, both change together.
OUTPUT_TOOLS: frozenset[str] = frozenset({"create_canvas", "report_gap"})

# Session-state key holding {tool_name: calls_so_far} for the current turn.
# Reset by critique.reset_turn_state, which is the LoopAgent's own
# before_agent_callback and therefore the one hook in this system that fires
# exactly once per turn rather than once per refinement cycle. Counting per
# turn rather than per cycle is deliberate: a refinement cycle that re-runs
# the same three searches is precisely the waste being bounded, so the budget
# must not refill when the loop goes round again.
STATE_KEY = "tool_calls_this_turn"


def _refusal(used: int, breakdown: dict[str, int]) -> dict[str, Any]:
    spent = ", ".join(f"{name} x{n}" for name, n in sorted(breakdown.items()))
    return {
        "error": "tool_call_budget_exhausted",
        "detail": (
            f"You have used all {used} evidence-gathering tool calls available for this "
            f"turn ({spent}). No further evidence-gathering tool can be called for this "
            "question - not this one, not a different one. Answer now from what you have "
            "already retrieved. If it does not contain what was asked for, call report_gap "
            "with the fact and the source you checked, then say so plainly in your answer - "
            "report_gap gathers nothing new either. If this question asked for a report, "
            "document or code file, you may still call create_canvas to produce it - that "
            "formats what you have and gathers nothing new."
        ),
    }


def enforce_tool_budget(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """Count this turn's evidence calls in total; refuse every tool past the ceiling.

    Returning None lets the real tool run, which is the path every
    well-behaved turn takes. Returning a dict short-circuits the call.

    Tools in OUTPUT_TOOLS are exempt on both counts - they neither increment
    the counter nor can be refused by it, because they produce rather than
    retrieve and a spent search budget has no bearing on rendering what was
    already found.

    The per-tool breakdown is still recorded, because it is the useful thing
    to see in a trace and in the refusal text - but it is the TOTAL that is
    compared against the ceiling. That distinction is the whole fix: counting
    per tool leaves an unspent budget on every other tool, and a turn that
    cannot search the knowledge base again will use one rather than conclude.

    The counter is read, incremented and written back as a new dict rather
    than mutated in place, because ADK tracks state deltas by assignment -
    mutating the nested dict returned by `state.get(...)` would not always be
    recorded as a change.
    """
    # Output tools bypass the ceiling entirely - not counted, never refused.
    # Checked before the counter is even read, so a spent budget cannot block
    # the artefact that the turn was asked for (see OUTPUT_TOOLS above).
    if tool.name in OUTPUT_TOOLS:
        return None

    counts = dict(tool_context.state.get(STATE_KEY) or {})
    used = sum(counts.values())
    if used >= MAX_TOOL_CALLS_PER_TURN:
        return _refusal(used, counts)
    counts[tool.name] = counts.get(tool.name, 0) + 1
    tool_context.state[STATE_KEY] = counts
    return None
