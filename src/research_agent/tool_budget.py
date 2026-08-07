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

## report_gap as a second, independent stop condition

The numeric ceiling above bounds a turn that searches too much. It does
nothing for the turn measured on the 2026-08-06 decline baseline: the agent
calls `report_gap` correctly, confirming the authoritative source does not
have the fact, and then - nowhere near the five-call ceiling - goes and
checks a second, unauthorised source anyway and answers or pads from it. A
call ceiling cannot see that failure at all, because the turn was never over
budget; it made the wrong kind of call, not too many of them.

The instruction has said not to do this for a long time (step 2: "its not
having the answer IS the answer, and searching elsewhere for a substitute
produces a figure from somewhere that was never authoritative"), and ADR-0023
already measured that wording losing to behaviour at scale, the same
conclusion this module's own ceiling exists for. So `report_gap` gets the
same treatment: once it has been called, `enforce_tool_budget` refuses every
further evidence-gathering tool call for the rest of the turn, regardless of
how much of the five-call ceiling is unspent. `report_gap` is not "one more
call towards the total" - it is a second, independent gate that can close
before the numeric one ever would.

`report_gap` and `create_canvas` stay callable after this - `report_gap`
because a multi-part question can legitimately report more than one gap, and
`create_canvas` because a report turn that hit a gap on one section still has
to render the sections it did answer. Both are already exempt from the
numeric ceiling for the same reason (see OUTPUT_TOOLS below); this gate does
not touch that exemption, it only ever refuses evidence tools.

## declare_plan as a third, independent gate (2026-08-06)

The two gates above bound HOW MUCH a turn searches and WHEN it must stop.
Neither bounds WHICH tool it is allowed to search with, and that turned out
to be the actual shape of the decline defect: across the 32 turns in the
2026-08-06 baseline that called `report_gap`, an evidence tool that was never
authoritative for the fact ran BEFORE `report_gap` in 32 of 32 - the agent
consults a source that was never right for the question, and only then
reports that the right one is empty. Neither the numeric ceiling nor the
report_gap gate can see this, because the turn was never over budget and
report_gap was never called too late - the tool call itself was simply never
supposed to happen.

`src/tools/declare_plan.py` lets the model commit, once per turn (or amended,
see below), to which tool is authoritative for which fact. Once a plan has
been declared, `enforce_tool_budget` refuses any evidence tool whose name was
not named as a source anywhere in that plan - see `_plan_refusal` and
`declare_plan.py`'s module docstring for the full reasoning, including why an
undeclared source is refused rather than merely discouraged, why the model
may amend the plan before a declared source has actually been called but not
after, and why `sources` values must be the tool's runtime `tool.name`
(`web_search_agent`) rather than the name it goes by in the INSTRUCTION prose
(`web_search_tool`).

This gate fails open: a turn that never calls `declare_plan` is checked
against no plan at all, and every evidence tool behaves exactly as it did
before this gate existed. That is deliberate (see `declare_plan.py`) and its
cost is real - skipping the declaration is a complete escape from the gate -
so declaration uptake has to be read off every sweep, not assumed.

## The gate became fact-level (2026-08-07)

As first built, the third gate did less than ADR-0024 said it did. The 2026
-08-06 audit, findings 3 and 4, found two holes, and both are closed here.

`record_declared_plan` stored `sorted(set(sources))` and discarded `facts`
entirely, so the gate could only ask "is this tool named anywhere in the
plan". On any question with more than one fact that is nearly no constraint:
a plan declaring `get_financial_data` for the BTC price and `web_search_agent`
for background on an ETF approval let the price be answered from the web, and
the gate saw a declared tool and allowed it. Every declared tool was legal for
every fact.

The fix could not be to read the missing signal, because it does not exist -
`before_tool_callback` receives `(tool, args, tool_context)` and nothing in
any of them says which fact a call is serving (see `src/tools/fact_tag.py`
for what was checked in the installed ADK and why). So the binding is created
instead: every evidence tool now takes a `fact` argument naming the declared
fact it is gathering, and this module checks that fact's declared source
against the tool being called. `report_gap(fact, source_checked)` already
worked this way, so the convention is the project's own rather than new.

The second hole was the amendment lock. It computed the sources being DROPPED
and refused only those, so the model could keep an already-called source and
ADD another - and `_amendment_refusal`'s own text told it to. That is exactly
the "the right source came back empty, so try a different one" fallback step
2 of the INSTRUCTION forbids, laundered through a legal-looking amendment.
With fact-level state the lock can be precise instead of merely tight: an
amendment may add entirely new facts with any source, and may still re-plan
any fact that has not been executed yet, but a fact whose declared source has
actually run is fixed and can be neither re-pointed nor given a second
source. The remedy for that case is `report_gap`, which is what the refusal
now says.

What is still NOT enforced, stated because ADR-0024's original text
overstating this is itself an audit finding: this disciplines execution, not
planning judgement. A fact whose source is mis-declared from the outset and
then dutifully executed passes every gate here, because nothing in this
module knows which source SHOULD have been authoritative. That is routing's
remaining ceiling and no gate can reach it.
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
#
# declare_plan added 2026-08-06: it gathers nothing either, for the same
# reason create_canvas and report_gap don't count against the ceiling - see
# this module's docstring, "declare_plan as a third, independent gate".
#
# What breaks if this and schema.py's copy drift apart is asymmetric, and
# worth knowing before editing either. A tool added to schema.py only is
# exempt from the redundancy metric but still counted against
# MAX_TOOL_CALLS_PER_TURN, so a well-behaved turn is refused one call early.
# A tool added here only is exempt from the ceiling but still lands in
# tools_called, so every turn using it reads as a redundancy regression -
# which is the false signal ADR-0016 was written about.
OUTPUT_TOOLS: frozenset[str] = frozenset({"create_canvas", "report_gap", "declare_plan"})

# Session-state key holding {tool_name: calls_so_far} for the current turn.
# Reset by critique.reset_turn_state, which is the LoopAgent's own
# before_agent_callback and therefore the one hook in this system that fires
# exactly once per turn rather than once per refinement cycle. Counting per
# turn rather than per cycle is deliberate: a refinement cycle that re-runs
# the same three searches is precisely the waste being bounded, so the budget
# must not refill when the loop goes round again.
STATE_KEY = "tool_calls_this_turn"

# Session-state key: True once report_gap has been called during the current
# turn. Same turn-scoping requirement as STATE_KEY above and reset alongside
# it in critique.reset_turn_state - a flag that survived into the next turn
# would refuse that turn's very first evidence call for a gap reported
# against a different question entirely.
STATE_KEY_GAP_REPORTED = "report_gap_called_this_turn"

# Session-state key holding the current turn's declared plan, as a sorted
# list of the distinct tool names named anywhere in the plan's `sources` (see
# src/tools/declare_plan.py). Written by record_declared_plan below, once
# declare_plan itself has validated the call and returned "status": "ok" -
# not written here directly, because a before_tool_callback runs before the
# real tool executes and cannot yet know whether that call was even valid.
# Absent or empty means no plan was declared this turn, which is what makes
# the gate fail open (see this module's docstring). Same turn-scoping
# requirement as STATE_KEY and STATE_KEY_GAP_REPORTED, reset alongside them
# in critique.reset_turn_state.
STATE_KEY_DECLARED_SOURCES = "declared_plan_sources_this_turn"

# Session-state key holding this turn's plan at FACT level: a mapping of
# normalised fact text to the sorted list of tool names declared authoritative
# for it. Added 2026-08-07 for audit finding 3, which found the flat source
# set above enforces something weaker than ADR-0024 claimed - with two facts
# and two sources, every declared tool was legal for every fact, so a price
# declared to get_financial_data could be answered from web_search_agent and
# the gate saw nothing wrong.
#
# STATE_KEY_DECLARED_SOURCES is kept alongside it rather than replaced. It is
# still what decides the cheap "this tool is in no part of the plan at all"
# refusal, it is what makes the gate fail open when no plan exists, and its
# absence is the condition every other consumer already reads.
STATE_KEY_DECLARED_FACTS = "declared_plan_facts_this_turn"

# Session-state key holding {normalised fact: sorted list of tool names
# actually called for it} this turn. This is what makes the amendment lock
# precise: audit finding 4 found the old lock refused only DROPPING an
# already-called source, so the model could keep it and ADD another, which is
# exactly the substitute-another-source fallback step 2 of the instruction
# forbids - and _amendment_refusal's own text spelled the move out. Knowing
# which fact each call served is what lets an amendment add genuinely new
# facts freely while refusing any change to a fact whose source has run.
STATE_KEY_FACT_CALLS = "declared_plan_fact_calls_this_turn"

# The argument every evidence tool now carries, naming which declared fact
# the call is serving. Imported by nothing here on purpose - the two plain
# function tools spell it in their signatures and src/tools/fact_tag.py puts
# it into the two AgentTool declarations, so this is the reader's copy.
FACT_ARG = "fact"

# Every refusal this module returns carries this key. Added 2026-08-07 for
# audit finding 12: src/ui/app.py used to relabel refused tool calls by
# matching one specific `error` value, so the two shapes added after it were
# rendered as successful instant calls - a blocked web search showing as
# "Searching the web for 0.0s" in green. Re-enumerating the error values
# there would have reset the same trap for whoever adds a fifth.
#
# A marker key rather than a uniform response shape, because the four
# refusals are not interchangeable to the MODEL: three are tool errors it
# should act on ("error"/"detail") and _amendment_refusal is a rejected
# declare_plan call, which must keep declare_plan's own {"status": "error"}
# contract or the model reads it as a different kind of failure. The marker
# is additive to both shapes and invisible in neither.
REFUSAL_MARKER_KEY = "refused_by"
REFUSAL_MARKER = "tool_budget"


def _normalise_fact(fact: str) -> str:
    """Fold a fact string to the form the gate matches on.

    Case and whitespace only - nothing fuzzier. A model that paraphrases its
    own declared fact gets a refusal naming the declared facts verbatim,
    which it can correct in one call; fuzzy matching would instead bind the
    call to whichever fact scored highest and enforce the wrong source
    silently, which is the failure this whole gate exists to stop.
    """
    return " ".join(fact.casefold().split())


def _gap_refusal() -> dict[str, Any]:
    return {
        REFUSAL_MARKER_KEY: REFUSAL_MARKER,
        "error": "evidence_gathering_ended_by_report_gap",
        "detail": (
            "report_gap has already been called this turn. report_gap is the last "
            "evidence-gathering action a turn takes, so no further evidence-gathering "
            "tool can be called for this question - not this one, not a different one. "
            "Answer now from what you have already retrieved. For the fact report_gap "
            "recorded, name and cite the source you checked and write the prose decline "
            "(step 3) - do not search a different source for it. For any other part of "
            "the question, answer from what you already have. If this question asked for "
            "a report, document or code file, you may still call create_canvas to produce "
            "it - that formats what you have and gathers nothing new."
        ),
    }


def _amendment_refusal(locked: dict[str, list[str]]) -> dict[str, Any]:
    """Refuse an amendment that re-plans a fact whose declared source has run.

    `locked` maps each such fact to the tool name(s) already called for it.
    The wording has to keep the model from reading this as "declare_plan is
    broken": the amendment path is legitimate and stays open for every fact
    that has not been executed yet, and for facts it has not named before.
    """
    parts = [f"{fact!r} (already served by {', '.join(tools)})" for fact, tools in sorted(locked.items())]
    locked_text = "; ".join(parts)
    return {
        REFUSAL_MARKER_KEY: REFUSAL_MARKER,
        "status": "error",
        "detail": (
            f"This plan was not recorded. It changes the source for {len(locked)} fact(s) whose "
            f"declared source has already been called this turn: {locked_text}. Once a fact's "
            "declared source has actually run, that pairing is fixed - it can be neither dropped "
            "nor added to. Declaring the wrong source BEFORE calling it is a mis-plan and can "
            "still be corrected freely, and you may still add entirely new facts with any source. "
            "But a source that came back without the fact is a gap, not a mis-plan: call "
            "report_gap with that fact and that source instead of re-planning around it. To "
            "amend, re-send this plan with each fact above keeping exactly the source that "
            "already ran, or call report_gap now."
        ),
    }


def _plan_refusal(tool_name: str, declared: list[str]) -> dict[str, Any]:
    declared_text = ", ".join(declared)
    return {
        REFUSAL_MARKER_KEY: REFUSAL_MARKER,
        "error": "tool_not_in_declared_plan",
        "detail": (
            f"{tool_name} was not declared as a source in your plan (declare_plan), so this "
            f"call is refused. The source(s) your plan declared are: {declared_text} - call "
            "one of those for the outstanding work instead. If the declared source has "
            "already been called and did not contain the fact, that is the gap: call "
            "report_gap with the fact and that source, then write the prose decline - do not "
            "substitute an undeclared source instead. If your plan genuinely named the wrong "
            "source for this fact and you have not yet called it, call declare_plan again to "
            "amend the plan before trying this tool."
        ),
    }


def _fact_missing_refusal(tool_name: str, declared_facts: list[str]) -> dict[str, Any]:
    """Refuse a call whose `fact` names nothing in the declared plan.

    Covers both an absent argument and one that does not match. Listing the
    declared facts verbatim is the whole remedy: the model wrote them, so
    seeing them back is enough to copy one exactly, and that is cheaper than
    any matching heuristic that could bind the call to the wrong fact.
    """
    facts_text = "; ".join(repr(fact) for fact in declared_facts)
    return {
        REFUSAL_MARKER_KEY: REFUSAL_MARKER,
        "error": "fact_not_in_declared_plan",
        "detail": (
            f"This call to {tool_name} was refused because its `{FACT_ARG}` argument does not "
            f"match any fact in the plan you declared this turn. The facts you declared are: "
            f"{facts_text}. Call again with `{FACT_ARG}` copied exactly from that list. If this "
            "call is for something you did not plan for, that is a new fact - call declare_plan "
            "again to add it, with the source that is authoritative for it, and then make this "
            "call."
        ),
    }


def _fact_source_refusal(tool_name: str, fact: str, declared_sources: list[str]) -> dict[str, Any]:
    """Refuse a call to a tool that is not the declared source FOR THIS FACT.

    This is the refusal audit finding 3 exists for: the tool may well be
    declared somewhere in the plan, just not for the fact it is being called
    with, and the flat source set could not see the difference.
    """
    sources_text = ", ".join(declared_sources)
    return {
        REFUSAL_MARKER_KEY: REFUSAL_MARKER,
        "error": "tool_not_the_declared_source_for_this_fact",
        "detail": (
            f"This call to {tool_name} was refused. Your plan declared {sources_text} as the "
            f"authoritative source for {fact!r}, not {tool_name}. {tool_name} may be declared "
            "for a different fact in the same plan, but a source is authoritative per fact, not "
            f"for the whole question. Call {sources_text} for this fact instead. If "
            f"{sources_text} has already been called for it and did not contain the fact, that "
            "is the gap: call report_gap with the fact and that source, then write the prose "
            "decline - do not substitute a different source. If your plan genuinely named the "
            f"wrong source for {fact!r} and you have not yet called it, call declare_plan again "
            "to amend the plan before trying this tool."
        ),
    }


def _refusal(used: int, breakdown: dict[str, int]) -> dict[str, Any]:
    spent = ", ".join(f"{name} x{n}" for name, n in sorted(breakdown.items()))
    return {
        REFUSAL_MARKER_KEY: REFUSAL_MARKER,
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

    Tools in OUTPUT_TOOLS are exempt from the numeric ceiling - they neither
    increment the counter nor can be refused by it, because they produce
    rather than retrieve and a spent search budget has no bearing on
    rendering what was already found.

    report_gap additionally sets a second, independent gate (see this
    module's docstring, "report_gap as a second, independent stop
    condition"): once it has been called, every evidence-gathering tool is
    refused for the rest of the turn regardless of the numeric ceiling. That
    check runs before the numeric one so a turn that reported a gap early
    cannot spend the rest of its five-call ceiling on an unauthorised source.

    declare_plan sets a third, independent gate (see this module's docstring,
    "declare_plan as a third, independent gate"): once a plan has been
    declared, an evidence tool whose name is not in it is refused, also
    independent of the numeric ceiling. declare_plan's own call is
    intercepted here too, before the real tool runs, to enforce that an
    amendment cannot drop a source that has already been called this turn
    (see _amendment_refusal) - the real tool has no session state to check
    that against, so this callback is the only place that check can happen.

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
    # declare_plan is intercepted before it runs, not because it needs a
    # ceiling exemption (OUTPUT_TOOLS already covers that below) but because
    # only this callback can see which of its previously-declared sources
    # have already been called this turn - the real tool function takes no
    # ToolContext (see declare_plan.py) and cannot check that itself. A
    # rejected amendment short-circuits here; the real tool never runs, and
    # record_declared_plan (called from agent.py's after_tool_callback) never
    # sees this attempt because there is no tool_response for it.
    if tool.name == "declare_plan":
        executed = tool_context.state.get(STATE_KEY_FACT_CALLS) or {}
        if executed:
            new_facts = args.get("facts")
            new_sources = args.get("sources")
            if isinstance(new_facts, list) and isinstance(new_sources, list):
                # Only compare facts that have actually been served. A fact
                # still unexecuted may be re-planned freely (the mis-plan
                # correction step 2 of the INSTRUCTION explicitly sanctions),
                # and a fact absent from `executed` entirely is a new one,
                # which an amendment is always allowed to add.
                proposed: dict[str, set[str]] = {}
                for fact, source in zip(new_facts, new_sources):
                    if isinstance(fact, str) and isinstance(source, str):
                        proposed.setdefault(_normalise_fact(fact), set()).add(source)
                locked = {
                    fact: sorted(called)
                    for fact, called in executed.items()
                    if proposed.get(fact, set()) != set(called)
                }
                if locked:
                    return _amendment_refusal(locked)
        return None

    # report_gap sets the second gate and is otherwise unbounded (a
    # multi-part question can legitimately report more than one gap) - so
    # this branch returns before either exemption or ceiling logic runs.
    if tool.name == "report_gap":
        tool_context.state[STATE_KEY_GAP_REPORTED] = True
        return None

    # Output tools (create_canvas, declare_plan) bypass the ceiling entirely -
    # not counted, never refused. declare_plan already returned above, so
    # this only ever matches create_canvas in practice; it stays in
    # OUTPUT_TOOLS anyway so schema.py and this module keep agreeing on what
    # counts as non-evidence (see OUTPUT_TOOLS above). Checked before the
    # counter is even read, so a spent budget cannot block the artefact that
    # the turn was asked for.
    if tool.name in OUTPUT_TOOLS:
        return None

    # The report_gap gate: refuses every evidence tool once report_gap has
    # been called this turn, independent of how much numeric ceiling is left
    # unspent - see this module's docstring for why the numeric ceiling alone
    # cannot catch this failure shape.
    if tool_context.state.get(STATE_KEY_GAP_REPORTED):
        return _gap_refusal()

    # The declared-plan gate: refuses an evidence tool that no declared fact
    # named as its source. Fails open when declared is empty (no plan was
    # ever recorded this turn) - see this module's docstring, "declare_plan
    # as a third, independent gate", and declare_plan.py for why that is a
    # deliberate property rather than an oversight.
    declared = tool_context.state.get(STATE_KEY_DECLARED_SOURCES) or []
    if declared and tool.name not in declared:
        return _plan_refusal(tool.name, declared)

    # The fact-level half of the same gate (audit finding 3). Runs only when a
    # plan exists, so the fail-open property above is unchanged: a turn that
    # never declared is never asked which fact it is serving.
    #
    # Order matters. The source-set check above answers "is this tool in the
    # plan at all", which is the cheaper and more obvious mistake, and its
    # refusal names the whole declared set. Only once the tool is somewhere in
    # the plan is it worth telling the model it is in the wrong PART of it -
    # the two refusals say different things and swapping them would answer a
    # question the model was not asking.
    declared_facts: dict[str, dict[str, Any]] = tool_context.state.get(STATE_KEY_DECLARED_FACTS) or {}
    if declared_facts:
        fact = args.get(FACT_ARG)
        entry = declared_facts.get(_normalise_fact(fact)) if isinstance(fact, str) else None
        if not entry:
            return _fact_missing_refusal(
                tool.name, [e["text"] for e in declared_facts.values()]
            )
        if tool.name not in entry["sources"]:
            return _fact_source_refusal(tool.name, entry["text"], entry["sources"])

    counts = dict(tool_context.state.get(STATE_KEY) or {})
    used = sum(counts.values())
    if used >= MAX_TOOL_CALLS_PER_TURN:
        return _refusal(used, counts)
    counts[tool.name] = counts.get(tool.name, 0) + 1
    tool_context.state[STATE_KEY] = counts

    # Record which fact this call served, for the amendment lock above. Only
    # ALLOWED calls are recorded, and only past the ceiling check, so a
    # refused call never locks a fact's source - a fact whose only attempt was
    # refused is still freely re-plannable, which is right: nothing was
    # actually consulted for it.
    if declared_facts:
        fact_calls = {name: list(tools) for name, tools in (tool_context.state.get(STATE_KEY_FACT_CALLS) or {}).items()}
        served = fact_calls.setdefault(_normalise_fact(args[FACT_ARG]), [])
        if tool.name not in served:
            served.append(tool.name)
        tool_context.state[STATE_KEY_FACT_CALLS] = fact_calls
    return None


def record_declared_plan(tool_context: ToolContext, tool_response: dict[str, Any]) -> None:
    """Persist a successfully-validated declare_plan call into this turn's gating state.

    Called from research_agent's after_tool_callback (see agent.py) once
    declare_plan itself has returned - never from the before_tool_callback
    above, because at that point the real tool has not run yet and may still
    reject the call (mismatched list lengths, an unknown source name; see
    declare_plan.py). Only a "status": "ok" response updates the declared
    plan, so an invalid declare_plan call leaves the previous plan (or no
    plan) in force rather than clobbering it with something that failed
    validation.

    Deliberately replaces STATE_KEY_DECLARED_SOURCES with the union of
    sources from THIS call, not the union with the previous plan: a later
    declare_plan call is read as the model's corrected, complete plan, not an
    addition to the old one. The lock in enforce_tool_budget's before_tool_callback
    is what stops that replacement being used to drop an already-called
    source - by the time this runs, that check has already passed.
    """
    if not isinstance(tool_response, dict) or tool_response.get("status") != "ok":
        return
    sources = tool_response.get("sources")
    facts = tool_response.get("facts")
    if not isinstance(sources, list) or not isinstance(facts, list):
        return
    tool_context.state[STATE_KEY_DECLARED_SOURCES] = sorted({s for s in sources if isinstance(s, str)})

    # The fact-level plan (audit finding 3). `facts` and `sources` are
    # parallel lists that declare_plan has already validated as equal length,
    # so zip cannot silently truncate a real plan here - but it is read
    # defensively anyway, because this runs on whatever the tool returned and
    # a malformed plan must leave the gate coherent rather than half-written.
    #
    # A fact is allowed to name more than one source. declare_plan's own
    # contract is one source per fact, but a model can legitimately list the
    # same fact twice with different sources when a fact genuinely needs
    # combining evidence (step 1 of the INSTRUCTION sanctions exactly that),
    # and collapsing those to one would refuse the second call for a plan the
    # tool itself accepted.
    # Each entry keeps the fact's ORIGINAL text alongside its sources. The
    # key has to be the normalised form so a call can match it, but a refusal
    # has to quote the model its own words back - echoing "btc price" at a
    # model that wrote "BTC price" invites it to "fix" the casing instead of
    # copying the fact.
    fact_plan: dict[str, dict[str, Any]] = {}
    for fact, source in zip(facts, sources):
        if isinstance(fact, str) and isinstance(source, str):
            entry = fact_plan.setdefault(_normalise_fact(fact), {"text": fact, "sources": []})
            if source not in entry["sources"]:
                entry["sources"].append(source)
    tool_context.state[STATE_KEY_DECLARED_FACTS] = fact_plan
