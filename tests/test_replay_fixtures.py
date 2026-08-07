"""The record/replay fixture layer: audit finding 6, and the `fact` decision that followed it.

Two things are pinned here, both from `src/evaluation/replay.py`:

1. `_patch_tools` must never patch anything in `schema.CONTROL_TOOLS` or
   `schema.OUTPUT_TOOLS`. Before this fix it excluded only `CONTROL_TOOLS`
   (`exit_loop`), so `declare_plan` and `report_gap` - both plain functions,
   both added after every one of the 15 stored fixture files was recorded -
   were patched like any evidence tool. In replay mode that means the very
   first call of every turn (`declare_plan`) hit an empty fixture and raised
   `FixtureMissError` immediately: the deterministic replay tier was entirely
   unavailable, not just stale.

   The test drives this against `schema.NON_EVIDENCE_TOOLS`
   (`CONTROL_TOOLS | OUTPUT_TOOLS`) and the real tool list registered on
   `root_agent`, not a hand-written list of names - so a tool added to either
   set later is covered automatically, which is exactly how this class of gap
   got missed the first time (a new tool existed; the exclusion list did not
   know about it).

2. Every evidence tool now carries a `fact` argument (ADR-0027) that the real
   tool never reads - `search_documents` and `get_financial_data` build their
   real behaviour from `query`/`category` alone. `fact` is model-generated
   prose, reworded between a recording run and a replay run the same way a
   free-text search query is, so leaving it in the fixture lookup key would
   silently degrade calls whose real inputs recur exactly (measured in this
   repo's own pre-ADR-0027 fixtures: `get_financial_data`'s `category` is a
   closed vocabulary - "crypto", "currencies", "stocks" - that repeats
   identically run to run) into merely-inexact matches, for a value that
   plays no part in what the tool actually does. The decision made in
   `_normalise_args` is to strip `fact` before the key is built, on both
   sides of the comparison. Pinned here directly against `_FixtureStore`,
   the real replay path, rather than against `_normalise_args` in isolation,
   so a future refactor that moves the stripping elsewhere still has to keep
   the externally-visible behaviour: identical real inputs with a differently
   worded `fact` still exact-match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation import replay
from src.evaluation.replay import FixtureSession, ToolFixtureSweep, _FixtureStore, _iter_llm_agents
from src.evaluation.schema import CONTROL_TOOLS, NON_EVIDENCE_TOOLS, OUTPUT_TOOLS
from src.research_agent.agent import root_agent

# --- _patch_tools must never touch CONTROL_TOOLS or OUTPUT_TOOLS ------------


def _snapshot_tools() -> dict[str, Any]:
    """Every plain-function tool currently on `root_agent`, keyed by name.

    AgentTool entries (web_search_agent, news_agent) are skipped: they are
    never in CONTROL_TOOLS or OUTPUT_TOOLS, and `_patch_tools` patches them by
    replacing `tool.run_async` in place rather than swapping the list entry,
    so an identity check on the tool object itself would not tell patched
    apart from unpatched for that branch anyway.
    """
    snapshot: dict[str, Any] = {}
    for llm_agent in _iter_llm_agents(root_agent):
        for tool in llm_agent.tools:
            name = getattr(tool, "__name__", None)
            if name is not None:
                snapshot[name] = tool
    return snapshot


def test_non_evidence_tools_are_not_individually_hand_picked() -> None:
    """Guards the test below's own premise.

    If CONTROL_TOOLS and OUTPUT_TOOLS were ever emptied by mistake, the
    patch-exclusion test below would still pass (there would be nothing left
    to check), and that would be a silent hole rather than a red test. This
    at least pins that both sets are non-empty and disjoint, which is what
    replay.py's `NON_EVIDENCE_TOOLS` import assumes.
    """
    assert CONTROL_TOOLS, "exit_loop must still be excluded"
    assert OUTPUT_TOOLS, "create_canvas/report_gap/declare_plan must still be excluded"
    assert not (CONTROL_TOOLS & OUTPUT_TOOLS)
    assert NON_EVIDENCE_TOOLS == CONTROL_TOOLS | OUTPUT_TOOLS


def test_patch_tools_never_touches_control_or_output_tools() -> None:
    """The audit finding 6 regression test.

    Patches `root_agent`'s real, live tool list (see replay.py's module
    docstring point 1 for why that mutation-in-place is required at all) and
    checks every name found in schema.NON_EVIDENCE_TOOLS - not a hard-coded
    list - is still the exact same object at the same index once the patch
    is installed. mode="record" is used only because ToolFixtureSweep.__enter__
    needs a mode other than "live" to call _patch_tools at all; entering the
    sweep does no file I/O by itself (that only happens inside `.run()`,
    which this test never calls), so nothing under data/eval/fixtures is
    touched.
    """
    before = _snapshot_tools()
    assert before, "root_agent must have at least one plain-function tool for this test to mean anything"

    non_evidence_present = {name for name in before if name in NON_EVIDENCE_TOOLS}
    assert non_evidence_present, "root_agent must carry at least one CONTROL_TOOLS/OUTPUT_TOOLS entry to test against"

    with ToolFixtureSweep(root_agent, mode="record"):
        after = _snapshot_tools()
        for name in non_evidence_present:
            assert after[name] is before[name], (
                f"{name} is in CONTROL_TOOLS or OUTPUT_TOOLS and must never be patched - "
                "a patched declare_plan/report_gap is exactly audit finding 6"
            )
        # The mirror check: every evidence tool DID get replaced with a wrapper.
        # Not the thing this test exists to catch, but without it a patch that
        # silently patched nothing at all would pass the loop above for the
        # wrong reason.
        evidence_present = {name for name in before if name not in NON_EVIDENCE_TOOLS}
        for name in evidence_present:
            assert after[name] is not before[name], f"{name} should have been wrapped for record/replay"

    # Patches must be fully undone once the sweep exits, regardless of mode -
    # ToolFixtureSweep.__exit__ runs every restorer.
    restored = _snapshot_tools()
    for name in before:
        assert restored[name] is before[name], f"{name} was not restored after the sweep exited"


# --- fact is excluded from the fixture lookup key ---------------------------


def _write_fixture(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(entries, sort_keys=True), encoding="utf-8")


async def _never_called() -> Any:
    raise AssertionError("real_call must not run in replay mode - the whole point of a fixture hit")


@pytest.mark.asyncio
async def test_fact_wording_alone_still_exact_matches(tmp_path: Path) -> None:
    """The decision, pinned against the real replay path.

    One recorded call to get_financial_data with a `fact` and a `category`.
    Replaying it with the SAME category but a DIFFERENT fact must still be an
    EXACT match (not the same-tool fallback) - because `fact` plays no part
    in what the tool actually does, and this is precisely the case that
    motivated stripping it: before ADR-0027 added `fact` to every call,
    `category`'s closed vocabulary already exact-matched here.
    """
    fixture_path = tmp_path / "fin-crypto.json"
    _write_fixture(
        fixture_path,
        [
            {
                "tool": "get_financial_data",
                "args": {"fact": "the current BTC price", "category": "crypto"},
                "response": {"data": "recorded response", "source": "https://example.test"},
            }
        ],
    )

    store = _FixtureStore(mode="replay", path=fixture_path)
    session = FixtureSession(mode="replay")

    result = await store.handle(
        "get_financial_data",
        {"fact": "worded completely differently this run", "category": "crypto"},
        _never_called,
        session,
    )

    assert result == {"data": "recorded response", "source": "https://example.test"}
    assert session.inexact_matches == [], (
        "a fact reworded between record and replay must not even register as an inexact match - "
        "the real inputs (category) were identical, so this must be the exact-match branch"
    )
    assert session.misses == []


@pytest.mark.asyncio
async def test_a_real_input_change_is_still_only_an_inexact_match(tmp_path: Path) -> None:
    """The control case: stripping `fact` must not make matching fuzzy for everything.

    Same fixture as above, but this call changes `category` - one of the
    tool's REAL inputs - not just `fact`. That must NOT exact-match; it should
    fall to the same-tool fallback (and be counted as inexact), exactly as any
    other genuinely different call to the same tool already does.
    """
    fixture_path = tmp_path / "fin-crypto.json"
    _write_fixture(
        fixture_path,
        [
            {
                "tool": "get_financial_data",
                "args": {"fact": "the current BTC price", "category": "crypto"},
                "response": {"data": "recorded response", "source": "https://example.test"},
            }
        ],
    )

    store = _FixtureStore(mode="replay", path=fixture_path)
    session = FixtureSession(mode="replay")

    result = await store.handle(
        "get_financial_data",
        {"fact": "the current BTC price", "category": "stocks"},
        _never_called,
        session,
    )

    assert result == {"data": "recorded response", "source": "https://example.test"}
    assert len(session.inexact_matches) == 1, "category differs for real, so this must be the fallback path"
    assert session.misses == []


def test_normalise_args_drops_fact_but_nothing_else() -> None:
    """Direct unit check on the building block the two tests above exercise end to end."""
    with_fact = replay._normalise_args({"fact": "one wording", "category": "crypto"})
    with_different_fact = replay._normalise_args({"fact": "a completely different wording", "category": "crypto"})
    with_different_category = replay._normalise_args({"fact": "one wording", "category": "stocks"})
    without_fact_at_all = replay._normalise_args({"category": "crypto"})

    assert with_fact == with_different_fact == without_fact_at_all, (
        "fact must have no influence on the key at all, regardless of its wording or its presence"
    )
    assert with_fact != with_different_category, "a real tool input must still change the key"
