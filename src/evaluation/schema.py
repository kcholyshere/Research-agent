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
    # Phase 5's A2A delegation, which is what NEWS_AGENT above was reserved for.
    #
    # "news_agent", NOT "get_latest_news". This entry has already been wrong
    # once, in exactly the way the web_search_agent comment above warns about.
    # The first phase 5 implementation was a plain function tool, so the name
    # came from `__name__` and "get_latest_news" was right. Rebuilding it on the
    # A2A protocol made the client a RemoteA2aAgent wrapped in an AgentTool, and
    # AgentTool names itself after the agent it wraps - so the name became the
    # remote agent's own, "news_agent". Same trap, same file, second occurrence.
    # Verified against the constructed tool object (src/tools/news_agent.py
    # exports NEWS_AGENT_TOOL_NAME read off the tool, not written as a literal)
    # and against a real turn's call.name.
    "news_agent": RouteTarget.NEWS_AGENT,
}

# Loop-control calls, not evidence-gathering. They show up in the same event
# stream as real tool calls and would otherwise be counted as routing.
CONTROL_TOOLS: frozenset[str] = frozenset({"exit_loop"})

# Output-producing calls, not evidence-gathering. Excluded from `tools_called`
# for the same reason CONTROL_TOOLS is, but the consequence is sharper here.
#
# `check_redundancy` compares len(tools_called) against the question's
# `max_tool_calls`, a bound written to police *searching*. create_canvas
# (phase 6) retrieves nothing - it renders facts already gathered - so
# counting it would add exactly one call to every artefact question and push
# each of them one over its own bound. Every one would read as a redundancy
# regression on a turn that behaved perfectly, and the bounds could not be
# raised to compensate without also loosening the real search budget they
# exist to enforce.
#
# Must stay in step with `src/research_agent/tool_budget.py`'s OUTPUT_TOOLS,
# which exempts the same names from the runtime ceiling. Two lists rather than
# one shared constant because the eval must be able to score a stored run
# without importing the agent (and therefore without triggering Langfuse
# instrumentation and a Vertex client) - but they are one concept, and a change
# to either is a change to both.
OUTPUT_TOOLS: frozenset[str] = frozenset({"create_canvas"})

# Everything that is not evidence-gathering. `tools_called` filters on this.
NON_EVIDENCE_TOOLS: frozenset[str] = CONTROL_TOOLS | OUTPUT_TOOLS


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

    # Alternative route sets that are equally correct, each a full substitute
    # for `expected_routes` rather than a per-route swap. Exists for exactly
    # one case so far: a question whose correct routing genuinely has more
    # than one right answer, because two different tools legitimately serve
    # the same half of the question and which one a live planner picks is not
    # itself a defect. `check_routing` passes if the routes used match
    # `expected_routes` OR any one set here, exactly - it does not mix and
    # match individual routes across sets. Empty for every other question, so
    # their routing semantics are unchanged.
    acceptable_routes: list[list[RouteTarget]] = field(default_factory=list)

    # Redundancy bound. Distinct from len(expected_routes): a question may
    # legitimately need two calls to one source (reformulation after a miss),
    # while five calls to it is the known premise-refuting defect.
    max_tool_calls: int = 2

    # True when the question asks for a deliverable rather than an answer, so
    # the turn is expected to call create_canvas and produce an artefact. The
    # format it should produce, when set, is one of canvas.OutputFormat -
    # asserting the format separately from the fact of an artefact existing,
    # because "produced a document when asked for code" is a distinct and more
    # interesting failure than "produced nothing".
    expects_artefact: bool = False
    artefact_format: str = ""
    # Strings the ARTEFACT must contain, checked separately from must_contain.
    # Kept apart on purpose: must_contain is scored against the answer, and an
    # artefact turn's answer is a short covering note while the substance lives
    # in the artefact. Folding them together would make it impossible to say
    # whether the facts reached the deliverable or only the reply.
    artefact_must_contain: list[str] = field(default_factory=list)

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

    # True when a genuinely unaddressed sub-question should make the critique
    # agent continue past cycle 1 - i.e. cycle_count is expected to be >= 2
    # under a budget that allows it. Exists for the open TODO on critique
    # calibration: every other critique_loop question is fully answerable, so
    # a critic that always exits scores identically to one that correctly
    # judged nothing was missing - there was no question in the set whose
    # correct behaviour was to NOT exit on cycle 1. This label is descriptive
    # only; nothing in metrics.py reads it yet. The record already carries
    # what a check would need (RunRecord.cycle_count, CycleRecord.
    # critique_outcome), so today this is read by hand from the stored run.
    # The smallest companion check would be something like
    # `check_critique_calibration`, asserting
    # `record.cycle_count >= 2 if question.expects_second_cycle else True`
    # (skipped entirely at budget 0, where no critique call happens at all) -
    # deliberately not added here, since metrics.py beyond check_routing is
    # out of scope for the change this field was written for.
    expects_second_cycle: bool = False

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
    # wasted_cycle, artefact.
    known_defects: dict[str, str] = field(default_factory=dict)

    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalQuestion:
        data = dict(raw)
        data["expected_routes"] = [RouteTarget(r) for r in data.get("expected_routes", [])]
        data["acceptable_routes"] = [
            [RouteTarget(r) for r in alt] for alt in data.get("acceptable_routes", [])
        ]
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

    # The artefact create_canvas rendered in this cycle, if any, taken from the
    # tool's function-response payload rather than from the agent's prose.
    #
    # Per cycle rather than per run because the refinement loop can render more
    # than once: a critique that raises a genuine gap on an artefact turn sends
    # the agent back, and the second pass rewrites the whole document. Storing
    # only the final one would hide exactly the case worth seeing - two full
    # renders where one would have done.
    #
    # Read from the function response, not the answer text, because the answer
    # on an artefact turn is a short covering note. Both `artefact_format` and
    # `artefact_path` come from the same payload, so a run can be checked for
    # "asked for HTML, produced markdown" offline with no agent call.
    artefact: str = ""
    artefact_format: str = ""
    # The code artefact's language, "" for markdown and html. Recorded so a
    # stored run can answer "asked for SQL, produced Python" offline - that is a
    # distinct defect from producing nothing, and without this field it is
    # simply invisible.
    artefact_language: str = ""
    artefact_path: str = ""


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

    # Set when this run shared the machine with other runs (the concurrent
    # phase of a sweep - see run_eval.run_sweep). Its assertions are unaffected
    # by contention, but its `latency_s` is not comparable with a run measured
    # alone, so metrics.summarise_latency segments on this rather than blending
    # the two populations. Runs in the sequential phase carry False, which is
    # also what every pre-concurrency run file deserialises to.
    contended: bool = False

    # Set when the run died on a Vertex quota rejection rather than on
    # anything the agent did. Concurrency makes these likely for the first
    # time (N concurrent turns are more than N concurrent Vertex requests -
    # research, critique and web_search_agent each call the model), and
    # without a dedicated flag they land in `error` and read as a genuine
    # regression. Same reasoning as `fixture_incomplete` directly above: a run
    # that tells you nothing must be excluded from aggregation and counted
    # separately, and a string prefix on `error` is the kind of cross-module
    # coupling that stops working the moment someone rewords a message.
    rate_limited: bool = False

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
            if name not in NON_EVIDENCE_TOOLS
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
    def artefacts(self) -> list[CycleRecord]:
        """Cycles that rendered an artefact, in order."""
        return [c for c in self.cycles if c.artefact]

    @property
    def artefact(self) -> str:
        """The artefact this turn ended with, or "" if it produced none.

        The LAST one rather than the first: when the refinement loop sends the
        agent back, the second render is the turn's output and the first is
        superseded. `artefacts` above keeps both, which is what makes a
        redundant re-render visible.
        """
        rendered = self.artefacts
        return rendered[-1].artefact if rendered else ""

    @property
    def artefact_format(self) -> str:
        rendered = self.artefacts
        return rendered[-1].artefact_format if rendered else ""

    @property
    def artefact_language(self) -> str:
        rendered = self.artefacts
        return rendered[-1].artefact_language if rendered else ""

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
