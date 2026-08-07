"""Regression tests for agent_docs/audit.md finding 12.

`src/ui/app.py` relabels a tool call the tool-call budget refused - so the
step trace does not read "Searching the web for 0.0s" in green, which looks
exactly like a search that ran and found nothing. The relabelling used to
match a single literal, `tool_call_budget_exhausted` - `tool_budget._refusal`'s
own "error" string, and the only one it knew about. `tool_budget.py` has
since grown two more refusal builders (`_gap_refusal`, `_plan_refusal`), each
returning its own "error" value the old check never learned, so calls the
report_gap gate or the declared-plan gate blocked rendered as ordinary,
successful, instant tool calls instead.

The fix in `src/ui/app.py._is_budget_refusal` recognises the response SHAPE
those three builders share (`{"error": <non-empty str>, "detail": <non-empty
str>}`) rather than any one of their string values, specifically so this
suite does not have to enumerate them either - the same trap, one level up.
What it deliberately proves instead:

1. Every refusal builder `tool_budget.py` has TODAY is exercised directly and
   checked against the discriminator - three are recognised, and the fourth
   (`_amendment_refusal`) is asserted NOT to be, with the reason pinned so the
   gap is a documented decision, not a silent one.
2. The exact set of refusal-builder names in `tool_budget.py` is pinned, so a
   fifth function named like the existing four (`..._refusal`) fails
   `test_tool_budget_refusal_builders_are_pinned` immediately. This is a
   naming-convention check, not a semantic one: a fifth gate added to
   `enforce_tool_budget` that returns its refusal inline, or under a
   differently-named helper, would not trip it - there is no way to discover
   "a new way `enforce_tool_budget` can short-circuit" from outside that
   function without driving it end-to-end for every gate, which is more than
   this fix's scope covers. Named here as a known limit of point 2, not
   claimed away.
3. Two tools' own domain errors, real ones from calling the real functions
   (not hand-built stand-ins), are proved NOT to be mislabelled as refusals -
   the specific new defect the task that produced this fix called out by name.
4. `_STEP_LABELS`/`_STEP_COLOURS` coverage is checked against
   `research_agent`'s actual registered tool list, not a hardcoded copy of it -
   so a sixth tool added to `agent.py` without a label fails here too, the
   same staleness shape as the refusal-shape defect above.
5. `app._ARTEFACT_MIME` is checked against `canvas.OutputFormat`'s actual
   `Literal` args, the same fixed-copy-of-another-module's-enum shape as
   point 4, just with a softer failure mode: a missing entry downloads with
   the wrong MIME type (`.get(fmt, "text/plain")`) rather than rendering
   visibly wrong, which is easy to miss in a demo.

Not fixed here, reported instead: `app._HIDDEN_STEPS` (`{"exit_loop"}`) has
the same shape again - a fixed set standing in for "critique_agent's internal
bookkeeping tools" - but critique_agent has exactly one tool today, and there
is no marker on a tool object this project controls that would let a test
derive "is this tool internal bookkeeping" without hardcoding the same one
name back (unlike points 4 and 5, which have a real enum elsewhere to check
against).
"""

from __future__ import annotations

import asyncio
import inspect
from typing import get_args

import pytest

from src.research_agent import tool_budget
from src.research_agent.agent import research_agent
from src.tools import canvas
from src.tools.canvas import create_canvas
from src.tools.declare_plan import declare_plan
from src.tools.financial_data import get_financial_data
from src.tools.report_gap import report_gap
from src.ui import app

# --- 1 & 2: every refusal builder tool_budget.py has today, and the set itself ---

# Pinned so a fifth function named "..._refusal" in tool_budget.py fails this
# test rather than silently going unchecked - see the module docstring, point
# 2, including what this naming-convention check does NOT catch. If this
# fails because a new one was added: give it the same
# {"error": ..., "detail": ...} shape as the three below and it is
# automatically caught by `_is_budget_refusal`; if it needs the
# `{"status": "error", ...}` shape instead (like `_amendment_refusal`), see
# this file's docstring and `app._is_budget_refusal`'s own docstring for why
# that shape cannot be caught without a tool_budget.py change, and update
# both docstrings to match.
_KNOWN_REFUSAL_BUILDERS = frozenset(
    {
        "_gap_refusal",
        "_plan_refusal",
        "_refusal",
        "_amendment_refusal",
        "_fact_missing_refusal",
        "_fact_source_refusal",
    }
)


def test_tool_budget_refusal_builders_are_pinned() -> None:
    found = {
        name
        for name, _ in inspect.getmembers(tool_budget, inspect.isfunction)
        if name.endswith("_refusal")
    }
    assert found == _KNOWN_REFUSAL_BUILDERS, (
        f"tool_budget.py's refusal builders changed (found {sorted(found)}). "
        "A builder was added or removed - update _KNOWN_REFUSAL_BUILDERS here, "
        "add a case for it below, and re-check whether "
        "src/ui/app.py._is_budget_refusal still recognises it before assuming "
        "the UI's refusal relabelling still works."
    )


def test_gap_refusal_is_recognised() -> None:
    assert app._is_budget_refusal(tool_budget._gap_refusal())


def test_plan_refusal_is_recognised() -> None:
    assert app._is_budget_refusal(
        tool_budget._plan_refusal("search_documents", ["get_financial_data"])
    )


def test_numeric_ceiling_refusal_is_recognised() -> None:
    assert app._is_budget_refusal(
        tool_budget._refusal(5, {"search_documents": 3, "web_search_agent": 2})
    )


def test_amendment_refusal_is_recognised_too() -> None:
    """The shape no payload inspection could ever have caught.

    `_amendment_refusal` returns `{"status": "error", "detail": ...}`, which
    is byte-for-byte what `declare_plan`'s and `create_canvas`'s own
    input-validation errors return (see the tests below). While
    `_is_budget_refusal` matched on payload shape, catching this one would
    have meant mislabelling those two genuine domain errors as budget
    refusals - the same mislabelling this fix exists to prevent, aimed at a
    different pair of tools - so it was deliberately left uncaught. The
    marker key added to every builder in tool_budget.py is what closed it,
    and this assertion flipping from `not` to plain is the signal that it
    landed.
    """
    refusal = tool_budget._amendment_refusal({"ifc's fy24 net income": ["search_documents"]})
    assert app._is_budget_refusal(refusal)
    # The model-facing contract is unchanged: declare_plan's caller still
    # reads {"status": "error", "detail": ...}. The marker is additive.
    assert refusal["status"] == "error"
    assert refusal["detail"]


# --- 3: real domain errors and real ordinary results, not stand-ins ---------


def test_declare_plan_success_is_not_mislabelled() -> None:
    result = declare_plan(["IFC's FY24 net income"], ["search_documents"])
    assert result["status"] == "ok"
    assert not app._is_budget_refusal(result)


def test_report_gap_response_is_not_mislabelled() -> None:
    result = report_gap("headcount by country", "the IFC 2024 Annual Report")
    assert result["status"] == "recorded"
    assert not app._is_budget_refusal(result)


def test_declare_plans_own_validation_error_is_not_mislabelled() -> None:
    """A real mis-plan (mismatched list lengths), not the budget refusing it.

    Same `{"status": "error", "detail": ...}` shape as `_amendment_refusal`
    (see that test above) - proving the discriminator does not catch this one
    either is the other half of proving it is genuinely shape-based rather
    than accidentally keying off which tool was called.
    """
    result = declare_plan(["fact one", "fact two"], ["search_documents"])
    assert result["status"] == "error"
    assert not app._is_budget_refusal(result)


def test_create_canvas_own_validation_error_is_not_mislabelled() -> None:
    """create_canvas's own "you got the arguments wrong" response.

    Deliberately a mismatched-length request, which `CanvasRequest` rejects
    before any template renders or any file is written under data/ - this
    exercises the real validation path with no filesystem side effect.
    """
    result = create_canvas(
        title="Test artefact",
        output_format="markdown",
        section_headings=["Section A"],
        section_bodies=["Body A", "Body B"],
        citations=[],
    )
    assert result["status"] == "error"
    assert not app._is_budget_refusal(result)


@pytest.mark.asyncio
async def test_financial_data_own_domain_error_is_not_mislabelled() -> None:
    """get_financial_data's own error uses "error" but never "source"+"detail".

    An unknown category is rejected before any MCP network call, so this is
    safe to call directly rather than needing a live mcp-fetch service.
    """
    result = await get_financial_data("the current price of Bitcoin", "not-a-real-category")
    assert "error" in result
    assert "detail" not in result
    assert not app._is_budget_refusal(result)


def test_document_search_own_domain_error_shape_is_not_mislabelled() -> None:
    """search_documents wraps its own error in a LIST, not a dict.

    Built as a literal matching src/tools/document_search.py's real return
    shape (`[{"error": "..."}]`) rather than calling the real async function,
    which would need the FAISS index to be present or absent in a specific,
    controlled state - state this test must not disturb (a build is
    presently in progress in this checkout). The `isinstance(payload, dict)`
    check in `_is_budget_refusal` rules out anything list-shaped before the
    key check ever runs, so the outer list is what matters here, not the
    inner dict's own shape.
    """
    payload = [{"error": "The knowledge base index has not been built..."}]
    assert not app._is_budget_refusal(payload)


def test_ordinary_non_dict_and_empty_payloads_are_not_mislabelled() -> None:
    assert not app._is_budget_refusal(None)
    assert not app._is_budget_refusal("")
    assert not app._is_budget_refusal({})
    assert not app._is_budget_refusal({"detail": "no error key at all"})
    assert not app._is_budget_refusal({"error": "", "detail": "empty error string"})
    assert not app._is_budget_refusal({"error": "real error", "detail": ""})


# --- 4: label/colour coverage against the agent's actual registered tools --


def test_step_labels_and_colours_cover_every_registered_tool() -> None:
    """`_STEP_LABELS`/`_STEP_COLOURS` must cover research_agent's real tool list.

    Reads `research_agent.canonical_tools()` (the same resolved list ADK
    itself calls the agent with - verified against installed google-adk
    2.5.0) rather than a hardcoded copy of the seven tool names, so a tool
    added to `agent.py`'s `tools=[...]` list without a matching UI label is
    caught here - the same staleness shape as declare_plan/report_gap being
    added to tool_budget.py without a matching case in the old refusal check.

    Synchronous test wrapped around `asyncio.run`, the same pattern
    conftest.py's own `_build_context` uses for a one-off coroutine with no
    dependency on an already-running loop.
    """

    async def _tool_names() -> set[str]:
        tools = await research_agent.canonical_tools()
        return {tool.name for tool in tools}

    names = asyncio.run(_tool_names())
    assert names, "research_agent registered no tools - this test's premise no longer holds"

    missing_labels = names - app._STEP_LABELS.keys()
    missing_colours = names - app._STEP_COLOURS.keys()
    assert not missing_labels, f"tool(s) with no _STEP_LABELS entry: {sorted(missing_labels)}"
    assert not missing_colours, f"tool(s) with no _STEP_COLOURS entry: {sorted(missing_colours)}"


# --- 5: the same staleness check for canvas's own format enum --------------


def test_artefact_mime_covers_every_canvas_output_format() -> None:
    """`_ARTEFACT_MIME` must cover every format `canvas.OutputFormat` declares.

    Reads the `Literal`'s actual args via `typing.get_args` rather than a
    hardcoded copy of "markdown", "html", "code", so a fourth Canvas format
    added to `canvas.OutputFormat` without a matching MIME entry here fails
    this test instead of quietly downloading with the wrong content type.
    """
    formats = set(get_args(canvas.OutputFormat))
    assert formats, "canvas.OutputFormat declared no formats - this test's premise no longer holds"

    missing = formats - app._ARTEFACT_MIME.keys()
    assert not missing, f"canvas output format(s) with no _ARTEFACT_MIME entry: {sorted(missing)}"
