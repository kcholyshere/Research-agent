"""The fact-level declared-plan gate: audit findings 3 and 4.

Every test here is one of the audit's own failure scenarios, driven through
the real `enforce_tool_budget` against a real `ToolContext` (see
`tests/conftest.py` for why that is possible offline). Nothing is mocked
except the `BaseTool` stand-in that carries a name, because a name is all the
callback reads off it.

The two scenarios that matter most, in the audit's words:

- Finding 3: `multi-web-and-financial` declares `facts=["BTC price",
  "background on the ETF approval"]` with `sources=["get_financial_data",
  "web_search_agent"]`. The model then answers the price from
  `web_search_agent`. The old gate saw the tool in the declared set and
  allowed it.
- Finding 4: the plan is `facts=["headcount by country"]`,
  `sources=["search_documents"]`. The knowledge base returns nothing. The
  model calls `web_search_agent`, is refused, and re-declares with
  `sources=["search_documents", "web_search_agent"]`. Nothing was dropped, so
  the old lock recorded the amendment and the web search ran.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from src.research_agent import tool_budget
from src.tools.declare_plan import declare_plan


class _NamedTool(BaseTool):
    """A stand-in carrying only what `enforce_tool_budget` reads: a name."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, description="test double")


def _declare(ctx: ToolContext, facts: list[str], sources: list[str]) -> dict[str, Any]:
    """Run a declare_plan call end to end, exactly as the agent does.

    Through the before_tool_callback (which owns the amendment lock), then
    the real tool, then `record_declared_plan` from the after_tool_callback -
    because the bug in finding 4 lived in the interaction between those
    three, not in any one of them.
    """
    refusal = tool_budget.enforce_tool_budget(_NamedTool("declare_plan"), {"facts": facts, "sources": sources}, ctx)
    if refusal is not None:
        return refusal
    response = declare_plan(facts, sources)
    tool_budget.record_declared_plan(ctx, response)
    return response


def _call(ctx: ToolContext, tool_name: str, fact: str) -> dict[str, Any] | None:
    """Attempt one evidence call. None means the real tool would have run."""
    return tool_budget.enforce_tool_budget(_NamedTool(tool_name), {"fact": fact, "query": "..."}, ctx)


# --- Finding 3: a declared tool is not legal for every fact ------------------

BTC = "BTC price"
ETF = "background on the ETF approval"


def test_the_audits_own_scenario_is_refused(tool_context: ToolContext) -> None:
    _declare(tool_context, [BTC, ETF], ["get_financial_data", "web_search_agent"])

    refused = _call(tool_context, "web_search_agent", BTC)

    assert refused is not None, (
        "web_search_agent answering the BTC price is finding 3 exactly - it is declared in the "
        "plan, but for the other fact"
    )
    assert refused["error"] == "tool_not_the_declared_source_for_this_fact"
    assert "get_financial_data" in refused["detail"]
    assert BTC in refused["detail"]


def test_the_same_tool_is_allowed_for_the_fact_it_was_declared_for(tool_context: ToolContext) -> None:
    """The other half: the gate must not simply refuse the second source."""
    _declare(tool_context, [BTC, ETF], ["get_financial_data", "web_search_agent"])

    assert _call(tool_context, "web_search_agent", ETF) is None
    assert _call(tool_context, "get_financial_data", BTC) is None


def test_a_fact_may_declare_two_sources_when_it_genuinely_needs_both(tool_context: ToolContext) -> None:
    """Step 1 of the INSTRUCTION sanctions combining evidence for one fact."""
    _declare(tool_context, [BTC, BTC], ["get_financial_data", "web_search_agent"])

    assert _call(tool_context, "get_financial_data", BTC) is None
    assert _call(tool_context, "web_search_agent", BTC) is None


def test_a_fact_the_plan_never_named_is_refused_with_the_declared_facts(tool_context: ToolContext) -> None:
    _declare(tool_context, [BTC], ["get_financial_data"])

    refused = _call(tool_context, "get_financial_data", "something else entirely")

    assert refused is not None
    assert refused["error"] == "fact_not_in_declared_plan"
    assert BTC in refused["detail"], "the remedy is the model seeing its own facts back"


def test_a_missing_fact_argument_is_refused(tool_context: ToolContext) -> None:
    _declare(tool_context, [BTC], ["get_financial_data"])

    refused = tool_budget.enforce_tool_budget(_NamedTool("get_financial_data"), {"category": "crypto"}, tool_context)

    assert refused is not None
    assert refused["error"] == "fact_not_in_declared_plan"


def test_fact_matching_ignores_case_and_whitespace_but_nothing_more(tool_context: ToolContext) -> None:
    _declare(tool_context, [BTC], ["get_financial_data"])

    assert _call(tool_context, "get_financial_data", "  btc   PRICE ") is None
    # A paraphrase is refused rather than guessed at - binding a call to the
    # nearest fact would enforce the wrong source silently.
    assert _call(tool_context, "get_financial_data", "the price of Bitcoin") is not None


def test_a_tool_absent_from_the_whole_plan_gets_the_other_refusal(tool_context: ToolContext) -> None:
    """The source-set check still runs first, and says something different."""
    _declare(tool_context, [BTC], ["get_financial_data"])

    refused = _call(tool_context, "news_agent", BTC)

    assert refused is not None
    assert refused["error"] == "tool_not_in_declared_plan"


def test_the_gate_still_fails_open_when_no_plan_was_declared(tool_context: ToolContext) -> None:
    """A turn that never declared is gated by nothing - deliberate, ADR-0024."""
    assert _call(tool_context, "web_search_agent", "anything at all") is None


# --- Finding 4: the amendment lock refuses addition, not just dropping -------

HEADCOUNT = "headcount by country"


def test_adding_a_source_after_the_declared_one_ran_is_refused(tool_context: ToolContext) -> None:
    """The audit's bypass, step by step."""
    _declare(tool_context, [HEADCOUNT], ["search_documents"])
    assert _call(tool_context, "search_documents", HEADCOUNT) is None  # comes back empty
    assert _call(tool_context, "web_search_agent", HEADCOUNT) is not None  # refused, as before

    amended = _declare(tool_context, [HEADCOUNT, HEADCOUNT], ["search_documents", "web_search_agent"])

    assert amended["status"] == "error", "keeping the called source and adding another was the bypass"
    assert "report_gap" in amended["detail"], "the refusal has to name the actual remedy"
    assert _call(tool_context, "web_search_agent", HEADCOUNT) is not None, "the amendment must not have landed"


def test_dropping_a_called_source_is_still_refused(tool_context: ToolContext) -> None:
    """The case the old lock did catch, kept so the rewrite did not lose it."""
    _declare(tool_context, [HEADCOUNT], ["search_documents"])
    _call(tool_context, "search_documents", HEADCOUNT)

    amended = _declare(tool_context, [HEADCOUNT], ["web_search_agent"])

    assert amended["status"] == "error"


def test_re_pointing_a_fact_that_has_not_run_yet_is_allowed(tool_context: ToolContext) -> None:
    """A mis-plan caught before executing is exactly what step 2 sanctions."""
    _declare(tool_context, [HEADCOUNT], ["search_documents"])

    amended = _declare(tool_context, [HEADCOUNT], ["web_search_agent"])

    assert amended["status"] == "ok"
    assert _call(tool_context, "web_search_agent", HEADCOUNT) is None


def test_adding_a_genuinely_new_fact_after_a_call_is_allowed(tool_context: ToolContext) -> None:
    """Precision, not tightness: the lock is per fact, not per turn."""
    _declare(tool_context, [HEADCOUNT], ["search_documents"])
    _call(tool_context, "search_documents", HEADCOUNT)

    amended = _declare(tool_context, [HEADCOUNT, BTC], ["search_documents", "get_financial_data"])

    assert amended["status"] == "ok"
    assert _call(tool_context, "get_financial_data", BTC) is None


def test_a_refused_call_does_not_lock_the_fact(tool_context: ToolContext) -> None:
    """Nothing was consulted, so nothing should be fixed."""
    _declare(tool_context, [HEADCOUNT], ["search_documents"])
    assert _call(tool_context, "news_agent", HEADCOUNT) is not None  # refused, never ran

    amended = _declare(tool_context, [HEADCOUNT], ["web_search_agent"])

    assert amended["status"] == "ok"


# --- The gate's own invariants ----------------------------------------------


def test_a_rejected_declaration_leaves_the_turn_ungated(tool_context: ToolContext) -> None:
    """Recorded because it is the gate's largest remaining escape hatch.

    `declare_plan` rejects an unknown source name and writes nothing, so the
    turn runs exactly as if it had never declared. audit.md:163 found the
    INSTRUCTION was steering the model straight into this by naming the web
    tool `web_search_tool` - not a value `declare_plan` accepts - while never
    stating the four it does. The instruction now names them; this test pins
    the underlying behaviour so the consequence stays visible.
    """
    rejected = _declare(tool_context, [BTC], ["web_search_tool"])

    assert rejected["status"] == "error"
    assert "web_search_agent" in rejected["detail"]
    assert _call(tool_context, "news_agent", BTC) is None, "no plan was recorded, so nothing is gated"


def test_the_instruction_names_the_source_strings_declare_plan_accepts() -> None:
    """audit.md:163 directly: the prose and the tool must agree on the names."""
    from src.research_agent.agent import INSTRUCTION
    from src.tools.declare_plan import EVIDENCE_TOOL_NAMES

    for name in EVIDENCE_TOOL_NAMES:
        assert f'"{name}"' in INSTRUCTION, f"the instruction never tells the model to use {name!r}"
    assert "web_search_tool" not in INSTRUCTION, (
        "web_search_tool is the Python variable name, not a value declare_plan accepts - naming "
        "it in model-facing prose is what made every declaration attempt fail silently"
    )


def test_output_tools_are_never_gated_by_a_plan(
    make_tool_context: Callable[[], ToolContext],
) -> None:
    """A plan must never be able to stop the turn recording a gap or rendering."""
    ctx = make_tool_context()
    _declare(ctx, [BTC], ["get_financial_data"])

    for name in sorted(tool_budget.OUTPUT_TOOLS):
        assert tool_budget.enforce_tool_budget(_NamedTool(name), {}, ctx) is None, name


@pytest.mark.parametrize(
    "builder",
    [
        lambda: tool_budget._gap_refusal(),
        lambda: tool_budget._plan_refusal("news_agent", ["search_documents"]),
        lambda: tool_budget._refusal(5, {"search_documents": 5}),
        lambda: tool_budget._amendment_refusal({"a fact": ["search_documents"]}),
        lambda: tool_budget._fact_missing_refusal("news_agent", ["a fact"]),
        lambda: tool_budget._fact_source_refusal("news_agent", "a fact", ["search_documents"]),
    ],
)
def test_every_refusal_carries_the_marker(builder: Callable[[], dict[str, Any]]) -> None:
    """So src/ui/app.py never has to enumerate refusal shapes again."""
    assert builder()[tool_budget.REFUSAL_MARKER_KEY] == tool_budget.REFUSAL_MARKER
