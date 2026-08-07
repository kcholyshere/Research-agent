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
fact's declared source has actually been called this turn, that PAIRING is
locked. A later `declare_plan` call may add entirely new facts with any
source, and may still re-point any fact it has not yet acted on, but for a
fact whose source has run it must re-send exactly that fact and exactly that
source. Neither dropping it nor adding a second source alongside it is
accepted.

The "adding" half of that rule was missing until 2026-08-07, and its absence
was audit finding 4. The lock computed only which sources were being DROPPED
and refused those, so the model could keep `search_documents` and add
`web_search_agent` beside it - nothing dropped, amendment recorded, web
search runs. `_amendment_refusal`'s own text was even telling it to, since it
said to keep the locked source and call `declare_plan` again. What made the
precise rule possible is that the gate now knows which fact each call served
(see below), so it can distinguish "a new fact needs a new source", which is
legitimate, from "this fact needs a second source now that the first came
back empty", which is the fallback dressed as a plan.

Locking is per fact rather than per turn on purpose. A turn that has executed
one fact must still be able to plan the next one freely; a lock scoped to the
turn would have refused that too, and would have been tight rather than
correct.

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

Note what a refused declaration costs, because it is this gate's largest
escape hatch and audit.md:163 found the INSTRUCTION walking straight into it.
A rejected call writes nothing, so the turn proceeds with no plan recorded
and every evidence tool ungated - indistinguishable from a turn that never
declared. The INSTRUCTION used to name the web tool `web_search_tool`
throughout, which is the Python variable name and not a value this tool
accepts, while never stating the four that are; it now names all four
explicitly.

## Facts are matched, so the model has to quote itself

Declaring the plan is only half of it. Each evidence tool also takes a `fact`
argument, and the gate matches it against the facts declared here to find
that fact's authoritative source. Matching is on case and whitespace only -
a paraphrase is refused with the declared facts quoted back, rather than
bound to whichever fact looks closest. Guessing would enforce the wrong
source silently, which is the failure this gate exists to prevent.
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

    Each source is authoritative for ITS fact, not for the whole question.
    Every evidence tool takes a `fact` argument as well as its own: pass the
    fact from this plan, copied exactly, and the tool you call must be the
    source you declared for that fact. Calling a tool with a fact you gave to
    a different source is refused even though that tool appears in this plan.

    If you realise a fact needs a different source than you first declared,
    call this again with the corrected plan - but only before you have
    actually called the source you first declared for that fact. Once a
    fact's declared source has been called, that pairing is fixed: you may
    neither drop it nor add a second source beside it. If it came back
    without the fact, call report_gap for it instead. Adding new facts, and
    re-pointing facts you have not yet acted on, stay available.

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
