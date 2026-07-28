"""Record schema for the evaluation pipeline - deliberately fixed before any
runner or metric is written, because everything else is built against it.

The evaluation plan (references/evaluation_brainstorm.md) makes one explicit
demand of this file: design the record with an iteration index and a routing
target enum from the start, so phases 4 and 5 do not force a rewrite of
stored run outputs. Both are here for that reason and not because today's
agent needs them - `RouteTarget.NEWS_AGENT` has nothing behind it until phase
5's A2A delegation exists, and a run stored today will still parse once it
does.

Why a per-cycle record rather than one row per run: phase 4's loop can run
the research agent more than once per turn, and the first failure mode the
plan wants caught is "a loop that burns three cycles to restate the same
answer". That is only detectable if each cycle is stored separately - an
aggregated run row cannot distinguish three cycles of real work from three
cycles of restatement. It also falls out of the event scan for free, since
the loop emits one final response per research cycle anyway.

Why dataclasses over Pydantic: these are written once and read once, with no
untrusted input anywhere, so validation buys little; plain dataclasses keep
the JSON round-trip obvious and add no dependency. If phase 6's Canvas work
brings Pydantic in for artefact validation, revisit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RouteTarget(str, Enum):
    """Where a question's evidence is supposed to come from.

    A str-valued Enum so `asdict()` serialises it as a readable string with no
    custom encoder, and a stored run stays legible without this module.

    NEWS_AGENT is unreachable today by design - phase 5 adds the A2A News
    Agent as a fourth routing target, and the plan calls for the dataset to
    carry delegation cases from the outset rather than being restructured
    later. NONE covers questions that should be answered (or declined) with
    no tool call at all, which is a real assertion: an agent that searches
    when it should decline is the exact defect this tier exists to catch.
    """

    KNOWLEDGE_BASE = "knowledge_base"
    WEB = "web"
    FINANCIAL = "financial"
    NEWS_AGENT = "news_agent"
    NONE = "none"


# The agent's tool names are an implementation detail of src/tools/; the
# dataset is written in terms of routing targets so that renaming a tool, or
# swapping google_search for Tavily (a live TODO), does not invalidate a
# stored dataset or run. Kept here rather than in the runner because both the
# runner and the metrics need it and neither should own it.
TOOL_TO_ROUTE: dict[str, RouteTarget] = {
    "search_documents": RouteTarget.KNOWLEDGE_BASE,
    # "web_search_agent", NOT "web_search_tool". The Python name in
    # src/tools/web_search.py is the AgentTool variable; ADK's AgentTool
    # names itself after the sub-agent it wraps (`super().__init__(name=
    # agent.name)`), so the name appearing in a real event's `call.name` is
    # the sub-agent's. Verified against the installed google-adk 2.5.0, after
    # the first version of this dict guessed the variable name and would have
    # silently reported zero routes used for every web question - the routing
    # assertion would have failed on a naming mismatch while looking like a
    # genuine routing defect.
    "web_search_agent": RouteTarget.WEB,
    "get_financial_data": RouteTarget.FINANCIAL,
}

# Loop-control calls, not evidence-gathering. They show up in the same event
# stream as real tool calls and would otherwise be counted as routing.
CONTROL_TOOLS: frozenset[str] = frozenset({"exit_loop"})


@dataclass
class EvalQuestion:
    """One labelled question in the evaluation set.

    The labels are assertions, not documentation: `expected_routes` is what
    the agent should consult, and anything it consults beyond that set is a
    routing error. `max_tool_calls` bounds redundancy separately, because
    calling the right tool five times is a different defect from calling the
    wrong tool once - the premise-refuting case does exactly the former.
    """

    id: str
    question: str
    expected_routes: list[RouteTarget]
    tags: list[str] = field(default_factory=list)

    # Redundancy bound. Distinct from len(expected_routes): a question may
    # legitimately need two calls to one source (reformulation after a miss),
    # while five calls to it is the known premise-refuting defect.
    max_tool_calls: int = 2

    # Content expectations. Kept deliberately weak - the plan is explicit that
    # anything touching live search or market data can carry routing and
    # latency assertions but never a fixed expected answer, because identical
    # code and an identical question gave 1 search one day and 5 the next
    # purely because "yesterday" moved.
    expects_citation: bool = True
    expects_decline: bool = False
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)

    # True when the correct answer changes with the news or the market. A
    # volatile question's content assertions only run in replay mode, where a
    # recorded fixture pins the world; in live mode it is scored on routing
    # and latency alone.
    volatile: bool = False

    # Assertions this question is expected to fail today against a known,
    # logged defect, as {assertion key: short defect reference}. The run still
    # records the failure - it is not suppressed - but it is reported apart
    # from regressions, so a known-broken case does not drown out a
    # newly-broken one.
    #
    # Keyed per assertion rather than per question on purpose: every web
    # question fails the citation check today (no web trace has yet produced a
    # real URL, only phrases like "(Google Search)"), but their routing is
    # currently correct and must stay able to break visibly. A question-level
    # flag would bucket a fresh routing regression as "known" and hide it,
    # which is the opposite of what this field exists for.
    #
    # Recognised keys: routing, redundancy, citation, decline, content,
    # wasted_cycle.
    known_defects: dict[str, str] = field(default_factory=dict)

    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalQuestion:
        data = dict(raw)
        data["expected_routes"] = [RouteTarget(r) for r in data.get("expected_routes", [])]
        return cls(**data)


@dataclass
class CycleRecord:
    """One pass of the research agent within a single turn.

    `index` is the iteration index the plan asks to be designed in from the
    start: 0 for the first research pass, 1 for the pass after a critique
    raised follow-ups, and so on. With a critique budget of 0 there is exactly
    one of these, which is what makes budget 0 the clean pre-phase-4 baseline
    arm.
    """

    index: int
    tools_called: list[str] = field(default_factory=list)
    draft: str = ""

    # What the critique agent did after this cycle. "skipped" is a real and
    # distinct outcome, not a missing value: critique.py short-circuits the
    # LLM call entirely for a spent budget or a financial-only cycle, and
    # "no critique call happened" is a stronger guarantee than "the model
    # approved quickly" - the eval should be able to tell them apart.
    critique_outcome: str = ""  # "exit" | "continue" | "skipped" | ""
    critique_followups: str = ""


@dataclass
class RunRecord:
    """One question, run once, against one arm.

    Deliberately a superset of ab_harness.RunResult rather than a replacement
    for it: that harness stays the quick two-arm comparison tool, and this is
    the persisted form. `latency_s` is wall-clock and stays authoritative -
    the plan notes that Langfuse nests tool spans inside the call_llm span
    that requested them, so span durations need child-subtraction before they
    mean anything, and that 18 of 47 traces returned empty on first fetch.
    `trace_id` is therefore recorded for offline attribution only; nothing in
    a run blocks on fetching it.
    """

    question_id: str
    question: str
    arm: str
    rep: int
    mode: str  # "live" | "replay"

    latency_s: float | None = None  # None when the run timed out
    timed_out: bool = False
    error: str | None = None

    # Set when a replay run aborted on a fixture miss. A dedicated field
    # rather than a prefix convention on `error`, because this flag decides
    # whether the run is admitted to metric aggregation at all - a
    # fixture-incomplete run tells you nothing and must be excluded - and a
    # string prefix shared across modules is the kind of coupling that
    # silently stops working the moment someone reworks an error message.
    fixture_incomplete: bool = False

    cycles: list[CycleRecord] = field(default_factory=list)
    answer: str = ""
    trace_id: str | None = None

    @property
    def tools_called(self) -> list[str]:
        """Every evidence-gathering tool call in the turn, in order."""
        return [
            name
            for cycle in self.cycles
            for name in cycle.tools_called
            if name not in CONTROL_TOOLS
        ]

    @property
    def routes_used(self) -> list[RouteTarget]:
        """Distinct routing targets consulted, order-preserved.

        Unknown tool names are dropped rather than raising: a tool added in a
        later phase should not make an older run unreadable.
        """
        seen: dict[RouteTarget, None] = {}
        for name in self.tools_called:
            route = TOOL_TO_ROUTE.get(name)
            if route is not None:
                seen.setdefault(route, None)
        return list(seen)

    @property
    def cycle_count(self) -> int:
        """How many research cycles this turn used.

        The plan requires latency to be segmented by cycle count rather than
        aggregated, because the agreed 15s target is scoped to single-tool
        turns with no refinement loop.
        """
        return len(self.cycles)


@dataclass
class EvalRun:
    """A complete pass of the question set over one or more arms."""

    settings: dict[str, Any]
    records: list[RunRecord] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> EvalRun:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = [
            RunRecord(
                **{**r, "cycles": [CycleRecord(**c) for c in r.get("cycles", [])]},
            )
            for r in raw.get("records", [])
        ]
        return cls(settings=raw.get("settings", {}), records=records)
