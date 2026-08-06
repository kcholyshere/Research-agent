"""The Report Gap Tool: makes "the authoritative source does not have this"
a first-class action, instead of a prose habit the model may or may not keep.

## Why this exists

Step 2 of `research_agent`'s INSTRUCTION already tells the model what to do
when the source that is authoritative for a fact does not contain it: "its
not having the answer IS the answer, and searching elsewhere for a substitute
produces a figure from somewhere that was never authoritative for the
question. Report the gap instead." Measured behaviour on `decline-*` and
`fin-coverage-gap` shows the words losing to the pull of "an empty result
feels like failure, and falling back to the web is the only move available
that feels like progress" - the model searches a second, unauthorised source
and answers from there instead of stopping.

`tool_budget.py`'s whole thesis is that a loop with no stop condition needs a
bound, not a more strongly worded request to stop - the same argument applies
here: the model had no explicit terminal action for "I checked, it is not
there", only the implicit option of continuing to search. This tool is that
action, so "declining" becomes something the model DOES rather than something
it merely writes.

## This is additive, not a replacement for the prose answer

Calling this tool does NOT end the turn and does NOT excuse the model from
writing the prose decline step 3 already requires - naming and citing the
source actually checked, exactly as it would for a fact it did find. This
tool only records that the gap was reported; the reader-facing answer still
has to say so. See the `detail` field of what this returns: it deliberately
restates that the next step is to write the prose decline, so continuing to
step 3 stays the obvious next move rather than something the model has to
independently remember to still do.

## Not an evidence source

Like `create_canvas` (see `src/tools/canvas.py`'s module docstring), this
tool gathers nothing - it records an outcome about evidence already gathered.
For that reason it is exempt from `tool_budget.MAX_TOOL_CALLS_PER_TURN` (see
`tool_budget.OUTPUT_TOOLS`) and from `check_redundancy` (see
`schema.OUTPUT_TOOLS`): a spent search budget must never be the reason the
model cannot call this, and a gap report must never itself count as one of
the wasteful calls the budget exists to police.
"""

from __future__ import annotations

from typing import Any


def report_gap(fact: str, source_checked: str) -> dict[str, Any]:
    """Record that a specific fact was checked and is not covered by the source that is authoritative for it.

    Call this when step 2's "Report the gap instead" applies: you consulted
    the single source that is authoritative for this fact (the knowledge base
    for IFC's own financial reporting, get_financial_data for market prices,
    and so on) and it does not contain what was asked. This is NOT a
    substitute for the source that has the fact simply returning nothing on a
    first try - reformulate and search again on that same source first, as
    step 2 already asks; call this once you have confirmed the gap, not
    before.

    This tool does not end the turn and does not write your answer for you.
    After calling it you still go on to step 3 and write the prose decline
    exactly as the instruction requires: name and cite the source you
    checked, say plainly that it does not cover this fact, and stop there -
    do not search a different source, and do not substitute a different
    period, a related figure, or "the closest available" number as if it
    answered the question.

    Args:
        fact: The specific fact that was not found, in plain terms - e.g.
            "IFC's headcount broken down by employee's home country in FY24".
        source_checked: The source that was checked and does not cover it,
            named the way you would cite it in your answer - e.g. "the IFC
            2024 Annual Report financial statements" or "get_financial_data's
            currency pairs".

    Returns:
        A dict with "status": "recorded" and a "detail" confirming the gap
        was logged and restating that your next step is the prose decline -
        it is not a message to show the reader as-is.
    """
    return {
        "status": "recorded",
        "detail": (
            f"Gap recorded: '{fact}' is not covered by {source_checked}. This does not "
            "end the turn - continue to your prose answer now. State plainly that "
            f"{source_checked} does not cover this, name and cite it exactly as you would "
            "any source you did find, and stop there. Do not search a different source, "
            "and do not substitute a different period, a related figure, or the closest "
            "available number as if it answered the question."
        ),
    }
