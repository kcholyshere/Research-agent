"""The Declared Plan Tool: makes the plan in step 1 of `research_agent`'s
INSTRUCTION a thing the executor is held to, instead of prose the model
writes and then may or may not follow.

## Why this exists

Step 1 already asks the planner to "break the question into the distinct
facts you need. For each, decide which single source is appropriate... State
the plan briefly." Measured on the 2026-08-06 decline baseline (40 runs, see
`agent_docs/decisions.md` ADR-0015 and ADR-0023), that plan changes nothing
about what step 2 is allowed to do: across the 32 turns that called
`report_gap`, an evidence tool that was never authoritative for the fact ran
BEFORE `report_gap` in 32 of 32. The plan is prose, so nothing enforces it.
This tool turns "State the plan briefly" into a call the executor can be
checked against: `src/research_agent/tool_budget.py`'s `enforce_tool_budget`
refuses an evidence tool whose name was not declared as a source for this
turn's plan.

This is `tool_budget.py`'s own thesis (a loop with no stop condition needs a
bound, not a firmer request to stop) applied one step earlier: ADR-0015
bounds how much a turn searches, ADR-0023's `report_gap` gives the model a
terminal action for "the authoritative source doesn't have it", and this tool
constrains WHICH tool a turn is allowed to search with in the first place.

## Not an evidence source

Like `create_canvas` and `report_gap`, this tool gathers nothing - it records
a decision about which tool is authoritative for which fact. It is exempt
from `tool_budget.MAX_TOOL_CALLS_PER_TURN` (see `tool_budget.OUTPUT_TOOLS`)
and from the redundancy metric for the same reason those two are: a spent
search budget must never be the reason the model cannot declare or amend its
plan, and declaring a plan must never itself count as one of the wasteful
calls the budget exists to police.

## Fail open when no plan was declared

If the model never calls this tool, evidence tools behave exactly as they did
before this change - `enforce_tool_budget` only checks a call against the
declared plan when a plan exists in this turn's state at all. A gate that
refused every evidence tool the moment the model skipped the declaration
would break every entrypoint the instant the model varied from the happy
path, which is a worse failure than the one this tool fixes. The cost is
real and is not assumed away: skipping the declaration is a complete escape
from the gate, so how often the model actually calls this tool is a number
that has to be measured on every sweep, not treated as given once the
mechanism exists.

## Amending the plan, and why that cannot become the bypass

Step 2 of the INSTRUCTION already draws this line: "Falling back to another
source is for when you planned the wrong one, not for when the right one
came back empty." A gate that refuses any undeclared source has to let the
model correct a genuine mis-plan, or it would also block the case the
instruction explicitly sanctions - realising before executing that the wrong
tool was named. So this tool may be called more than once in a turn, and a
later call replaces the plan `enforce_tool_budget` checks calls against.

What stops that turning into "the declared source came back empty, so
re-declare a different one and carry on" - exactly the fallback step 2
forbids - is enforced in `tool_budget.enforce_tool_budget`, not here: once a
declared source has actually been called this turn, it is locked. A later
`declare_plan` call may add new facts and sources freely, and may still
harmlessly re-list an already-called source, but it may not drop one that has
already run. Dropping a locked source is refused with a structured error
naming the source and pointing at `report_gap` - the tool that exists
specifically for "the right source came back empty" - instead of letting the
amendment stand in for it. So the only sources that can ever be
re-planned away from are ones that have not yet been tried, which is exactly
the "planned the wrong one" case and not the "came back empty" case.

## Source names must match the tools the executor actually calls, not their
## variable names in the INSTRUCTION prose

`sources` values are matched against `tool.name` at the point a real evidence
tool is called, which is NOT always the Python name a tool is imported under.
The web search AgentTool is invoked in the codebase as `web_search_tool`, but
`AgentTool` names itself after the agent it wraps, so the name that actually
appears on a call is `web_search_agent` - the exact trap
`src/evaluation/schema.py`'s `TOOL_TO_ROUTE` comment and `src/tools/news_agent.py`
both record having hit already. Use exactly these four names in `sources`:

- `search_documents` - the private knowledge base.
- `get_financial_data` - live market data.
- `web_search_agent` - the public internet (NOT `web_search_tool`).
- `news_agent` - the delegated News Agent.

Any other value is refused with a structured error naming the valid four, so
a wrong name is a correctable mistake rather than a plan that silently can
never be matched.
"""

from __future__ import annotations

from typing import Any

# The exact tool.name values `enforce_tool_budget` sees on a real evidence
# call - see the module docstring's note on web_search_agent vs
# web_search_tool. Kept here, not imported from src/evaluation/schema.py's
# TOOL_TO_ROUTE: that module is eval-owned and this is production code called
# on every turn, and ADR-0016 already established the precedent of two
# independent, commented constants over one shared import across that
# boundary (see OUTPUT_TOOLS in both tool_budget.py and schema.py).
EVIDENCE_TOOL_NAMES: frozenset[str] = frozenset(
    {"search_documents", "get_financial_data", "web_search_agent", "news_agent"}
)


def declare_plan(facts: list[str], sources: list[str]) -> dict[str, Any]:
    """Declare, before executing, which single tool is authoritative for each fact this question needs.

    Call this once, right after step 1's planning and before any evidence
    tool call, with one entry per fact you identified and the source you
    decided is authoritative for it. `facts` and `sources` are parallel
    lists of the SAME length: the Nth source is the tool authoritative for
    the Nth fact.

    Only a tool named in `sources` can be called for the rest of this turn -
    an evidence tool you did not declare will be refused. If you realise a
    fact needs a different source than you first declared, call this again
    with the corrected plan, but only before you have actually called the
    source you first declared for that fact: once a declared source has been
    called, it is locked and cannot be dropped by re-declaring - if it came
    back without the fact, call report_gap for it instead, the same as if
    you had never called this tool at all.

    Args:
        facts: The distinct facts this question needs, in plain terms - e.g.
            "IFC's FY24 net income", "the current USD/PLN exchange rate".
        sources: The tool authoritative for each fact, in the same order and
            the same number as `facts`. Each value must be exactly one of:
            "search_documents" (the private knowledge base), "get_financial_data"
            (live market data), "web_search_agent" (the public internet - note
            this is NOT the same string as "web_search_tool"), or "news_agent"
            (the delegated News Agent).

    Returns:
        On success, a dict with "status": "ok" and a "detail" confirming the
        plan is recorded. On invalid input, a dict with "status": "error" and
        a "detail" naming what to fix - correct it and call declare_plan
        again.
    """
    if len(facts) != len(sources):
        return {
            "status": "error",
            "detail": (
                f"facts and sources must be the same length - got {len(facts)} fact(s) and "
                f"{len(sources)} source(s). They are parallel lists: the Nth source is "
                "authoritative for the Nth fact."
            ),
        }
    if not facts:
        return {
            "status": "error",
            "detail": "facts and sources cannot both be empty - declare at least one fact and its source.",
        }
    unknown = sorted({s for s in sources if s not in EVIDENCE_TOOL_NAMES})
    if unknown:
        valid = ", ".join(sorted(EVIDENCE_TOOL_NAMES))
        return {
            "status": "error",
            "detail": (
                f"sources contains value(s) not recognised as a tool: {', '.join(unknown)}. "
                f"Valid values are exactly: {valid}. Note the public web source is "
                "\"web_search_agent\", not \"web_search_tool\"."
            ),
        }

    return {
        "status": "ok",
        "detail": (
            f"Plan recorded for {len(facts)} fact(s). Only {', '.join(sorted(set(sources)))} "
            "can be called as evidence tools for the rest of this turn - any other evidence "
            "tool call will be refused. Proceed to step 2 and execute this plan."
        ),
        "facts": facts,
        "sources": sources,
    }
