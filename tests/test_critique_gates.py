"""Regression tests for agent_docs/audit.md Findings 8 and 11b.

Finding 8: `critique._skip_critique_llm_call`'s financial-only shortcut
compared the cycle's tool set to `{"get_financial_data"}` by exact equality.
Once `declare_plan` became a mandatory tool on every turn (2026-08-06), a
single-fact financial turn always also called it, the cycle's tool set became
`{"declare_plan", "get_financial_data"}`, the equality stopped holding, and
the deterministic skip ADR-0010 built specifically to guarantee "no LLM call
happened here" for that path silently became unreachable - every financial
turn started paying for a full critique call again.

Finding 11b: `critique_agent` had no deadline check of its own.
`TURN_TIMEOUT_S` only gated `research_agent`'s before_agent_callback, so a
research cycle that overran to the full budget was still followed by a
critique call bounded only by `MODEL_CALL_TIMEOUT_MS` - the real worst case
for a turn was TURN_TIMEOUT_S plus one research cycle plus one full critique
call, not TURN_TIMEOUT_S.

Both fixes live in `_skip_critique_llm_call`, so both are exercised by
calling that function directly with a hand-built `CallbackContext` (see
conftest.py for why that is a faithful stand-in for the real thing) - no
`Agent`, `Runner`, or model call involved anywhere in this file.
"""

from __future__ import annotations

from unittest import mock

from google.adk.events.event import Event
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from src import config
from src.research_agent import critique, tool_budget, turn_deadline


def _tool_call_event(tool_name: str, author: str = "research_agent") -> Event:
    """A synthetic event standing in for one function-call turn of research_agent.

    `critique._tools_used_this_cycle` reads `event.get_function_calls()`, which
    in turn reads `content.parts[].function_call` (verified against the
    installed google-adk 2.5.0's `LlmResponse.get_function_calls`) - so a bare
    `Content` carrying one `FunctionCall` part is all the scan needs, with no
    tool_response, no model call, and no real tool object anywhere.
    """
    return Event(
        author=author,
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=tool_name))],
        ),
    )


def _seed_cycle(tool_context: ToolContext, tool_names: set[str]) -> None:
    """Append one synthetic research_agent event per tool name.

    Stands in for "the research cycle that just finished called exactly these
    tools" - `_tools_used_this_cycle` scans backwards from the most recent
    event to the previous critique_agent turn or the user message, so any
    author other than those two counts as part of "this cycle" for its
    purposes.
    """
    for name in tool_names:
        tool_context.session.events.append(_tool_call_event(name))


def _set_deadline_already_passed(tool_context: ToolContext) -> None:
    """Stamp this turn's deadline so it reads as already exceeded.

    Mirrors tests/test_turn_timeout.py's own pattern of shrinking
    `config.TURN_TIMEOUT_S` rather than sleeping or patching the clock: a
    negative budget makes `stamp_turn_deadline` write an absolute deadline
    that is already in the past the instant it is read, with no timing
    dependency on how fast the test itself runs.
    """
    with mock.patch.object(config, "TURN_TIMEOUT_S", -1.0):
        turn_deadline.stamp_turn_deadline(tool_context)


# --- Finding 8: the financial-only skip must survive declare_plan -----------


def test_financial_only_with_declared_plan_still_skips_deterministically(
    tool_context: ToolContext,
) -> None:
    """The audit's own failure scenario: declare_plan plus one financial lookup.

    Before the fix, `_tools_used_this_cycle(...) == {"get_financial_data"}`
    compared `{"declare_plan", "get_financial_data"}` to `{"get_financial_data"}`
    and found them unequal, so this cycle paid for a full critique call. The
    fix subtracts tool_budget.OUTPUT_TOOLS (minus create_canvas) before the
    comparison, so declare_plan no longer defeats it.
    """
    _seed_cycle(tool_context, {"declare_plan", "get_financial_data"})

    result = critique._skip_critique_llm_call(tool_context)

    assert result is not None, "expected the deterministic skip to fire - no LLM call should happen"
    assert tool_context.actions.escalate is True
    assert "financial figure" in result.parts[0].text
    # The skip must not also record a spent critique cycle - nothing was
    # actually critiqued.
    assert tool_context.state.get("critique_iterations_used", 0) == 0


def test_financial_only_with_report_gap_also_skips(tool_context: ToolContext) -> None:
    """report_gap is bookkeeping too (tool_budget.py) - it must not defeat the skip either.

    A financial fact that get_financial_data didn't have, declined via
    report_gap, is the same "nothing to critique" shape the financial-only
    skip already covers: the decline is deterministic, so there is no
    sub-question left for a critique call to usefully raise.
    """
    _seed_cycle(tool_context, {"declare_plan", "get_financial_data", "report_gap"})

    result = critique._skip_critique_llm_call(tool_context)

    assert result is not None
    assert tool_context.actions.escalate is True


def test_a_genuine_second_evidence_tool_does_not_skip(tool_context: ToolContext) -> None:
    """The other half of Finding 8's fix: a real second evidence tool must still defeat it.

    Subtracting the bookkeeping tools must not become so permissive that any
    cycle touching get_financial_data at all is treated as financial-only -
    search_documents here is a genuine second source of evidence, so this
    cycle has real content for the critique to review.
    """
    _seed_cycle(tool_context, {"declare_plan", "get_financial_data", "search_documents"})

    result = critique._skip_critique_llm_call(tool_context)

    assert result is None, "a second evidence tool must not take the financial-only skip"
    assert not tool_context.actions.escalate
    assert tool_context.state.get("critique_iterations_used") == 1


def test_create_canvas_defeats_the_financial_only_skip(tool_context: ToolContext) -> None:
    """The create_canvas decision this fix had to make explicitly.

    create_canvas is exempt from tool_budget's numeric ceiling for a
    different reason (it retrieves nothing) but it is NOT bookkeeping like
    declare_plan/report_gap - it renders the artefact ADR-0016 requires the
    critic to review. A financial-only cycle that also rendered a canvas must
    therefore still pay for a real critique call rather than take the
    deterministic skip.
    """
    _seed_cycle(tool_context, {"declare_plan", "get_financial_data", "create_canvas"})

    result = critique._skip_critique_llm_call(tool_context)

    assert result is None, "create_canvas must defeat the financial-only skip, not be subtracted like declare_plan"
    assert not tool_context.actions.escalate


def test_tools_with_nothing_to_critique_matches_output_tools_minus_canvas() -> None:
    """Pin the derivation itself, not just its effect on one scenario.

    Guards against a future edit that re-introduces a hand-written list (or
    forgets the create_canvas carve-out) without a single behavioural test
    happening to exercise the exact tool combination that would expose it.
    """
    assert critique._TOOLS_WITH_NOTHING_TO_CRITIQUE == tool_budget.OUTPUT_TOOLS - {"create_canvas"}
    assert "create_canvas" not in critique._TOOLS_WITH_NOTHING_TO_CRITIQUE
    assert "declare_plan" in critique._TOOLS_WITH_NOTHING_TO_CRITIQUE
    assert "report_gap" in critique._TOOLS_WITH_NOTHING_TO_CRITIQUE


# --- Finding 11b: critique_agent must share research_agent's turn deadline --


def test_a_turn_past_its_deadline_skips_the_critique_call(tool_context: ToolContext) -> None:
    """The audit's own failure scenario: the turn's whole-turn budget is already spent.

    Uses two genuine evidence tools (search_documents and web_search_agent) so
    neither the financial-only nor the budget-exhausted reason could
    coincidentally explain a skip - only the deadline check introduced for
    Finding 11b can make this fire.
    """
    _seed_cycle(tool_context, {"search_documents", "web_search_agent"})
    _set_deadline_already_passed(tool_context)

    result = critique._skip_critique_llm_call(tool_context)

    assert result is not None, "a turn past its deadline must not pay for a critique model call"
    assert tool_context.actions.escalate is True
    assert "time budget is already spent" in result.parts[0].text
    # Must not claim no answer was produced - research_agent already wrote a
    # real draft_answer this cycle, only the refinement pass is being skipped.
    assert "no answer" not in result.parts[0].text.lower()
    assert tool_context.state.get("critique_iterations_used", 0) == 0


def test_deadline_check_runs_before_budget_and_financial_checks(
    tool_context: ToolContext,
) -> None:
    """The deadline reason must win even when another skip reason would also apply.

    Not load-bearing for correctness (either reason alone already stops the
    LLM call), but pins the priority the docstring documents, so the message
    a trace shows stays informative about which of several possible reasons
    actually applied.
    """
    _seed_cycle(tool_context, {"declare_plan", "get_financial_data"})
    _set_deadline_already_passed(tool_context)

    result = critique._skip_critique_llm_call(tool_context)

    assert result is not None
    assert "time budget is already spent" in result.parts[0].text


def test_a_fresh_turn_within_its_deadline_is_unaffected(tool_context: ToolContext) -> None:
    """The other half: a turn well within budget must not be caught by the new check."""
    _seed_cycle(tool_context, {"search_documents", "web_search_agent"})
    turn_deadline.stamp_turn_deadline(tool_context)  # real TURN_TIMEOUT_S, freshly stamped

    result = critique._skip_critique_llm_call(tool_context)

    assert result is None
    assert not tool_context.actions.escalate


# --- Pre-existing behaviour that must survive both fixes untouched ----------


def test_budget_exhausted_still_skips(tool_context: ToolContext) -> None:
    tool_context.state["critique_budget"] = 0
    _seed_cycle(tool_context, {"declare_plan", "search_documents", "web_search_agent"})

    result = critique._skip_critique_llm_call(tool_context)

    assert result is not None
    assert tool_context.actions.escalate is True
    assert "budget for this request is spent" in result.parts[0].text
