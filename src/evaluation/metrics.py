"""Deterministic assertions and aggregation for the evaluation pipeline.

No LLM judge anywhere in this module, by design (references/evaluation_
brainstorm.md, ADR-0011): every defect this project has found so far -
routing, redundancy, repetition, citation - was an orchestration defect
visible in trace structure alone, and that is exactly what run_eval.py's
RunRecord/CycleRecord capture. A judge would add cost, latency and the
poor test-retest reliability the brainstorm explicitly rejected, for
assertions that do not need one.

Layout: one `check_*` function per assertion in the brief (routing,
redundancy, citation, decline, content, wasted_cycle), each returning an
`AssertionResult` or `None` when the assertion does not apply to this
question (e.g. a content check on a question with no must_contain/
must_not_contain). `evaluate_run` runs all of them for one (question, run)
pair and cross-references `question.known_defects`; `summarise` rolls a
whole `EvalRun`'s records up per arm.
"""

from __future__ import annotations

import difflib
import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable

from src.evaluation.schema import EvalQuestion, RunRecord

# The six assertion keys the brief fixes as the vocabulary for
# `EvalQuestion.known_defects`. Kept as a literal set (rather than inferring
# it from whatever check_* happens to run) so a typo'd key in questions.yaml
# fails to match anything rather than being silently accepted - the same
#"don't let a naming mismatch masquerade as a real result" concern schema.py
# already flags for TOOL_TO_ROUTE.
ASSERTION_KEYS = frozenset(
    {"routing", "redundancy", "citation", "decline", "content", "wasted_cycle"}
)

# 2026-07-28 measured baseline: 15s for single-cycle, single-tool turns only
# (references/evaluation_brainstorm.md, ADR-0010). Never applied to
# multi-cycle turns - their latency varies too much to hold to one number.
LATENCY_TARGET_S = 15.0

# Citation heuristics. The specific, measured defect (references/evaluation_
# brainstorm.md, "attribution is broken") is that web answers cite the
# literal phrase "(Google Search)" with no underlying URL - so the check has
# to positively distinguish a real source from that placeholder, not just
# test for the word "Source". KB answers cite a document path/name rather
# than a URL (see research_agent's synthesize instruction step 3), so a
# document-shaped reference counts as a real citation too.
_URL_RE = re.compile(r"https?://[^\s)\]]+")
_BROKEN_CITATION_RE = re.compile(r"\(Google Search\)", re.IGNORECASE)

# A filename was the original test for a document citation, and it was wrong:
# the agent never emits one. Measured on the first baseline, a KB answer
# citing "According to the IFC's 2024 Annual Report ... (Source: IFC Annual
# Report 2024 Financials, Consolidated Statements of Operations, page 64)"
# scored as "no citation of any kind", which made the citation assertion fail
# on essentially every question and buried the real defect in false
# positives. The corpus is a PDF on disk, but the agent cites it the way a
# person would, by name and page.
#
# So a document citation is now an attribution MARKER ("source:", "according
# to", "per") near a document NOUN ("annual report", "statement", "page 64").
# Both halves are required: the marker alone matches conversational filler
# ("according to the data"), and the noun alone matches an answer that merely
# mentions a report without attributing anything to it.
_DOC_FILENAME_RE = re.compile(r"[\w\-./]+\.(?:pdf|docx?|csv|json)\b", re.IGNORECASE)
_ATTRIBUTION_MARKER_RE = re.compile(
    r"\b(?:source|sources|according to|as (?:reported|stated) (?:in|by)|per|cited in|from the)\b",
    re.IGNORECASE,
)
_DOCUMENT_NOUN_RE = re.compile(
    r"\b(?:annual report|financial statements?|consolidated statements?|"
    r"balance sheet|filing|prospectus|report|financials|page\s+\d+|p\.\s*\d+)\b",
    re.IGNORECASE,
)

# Decline heuristics. Phrases the synthesize instruction's "say so plainly
# and stop" is expected to produce when neither source covers the question.
_DECLINE_PHRASES = (
    "does not contain",
    "do not contain",
    "not available",
    "not provided",
    "no information",
    "unable to find",
    "could not find",
    "cannot find",
    "not disclosed",
    "not specified",
    "does not provide",
    "does not state",
    "not covered",
    "no data",
    "not found",
)

# Padding threshold for the decline check. decline-headcount-by-country's
# known defect (questions.yaml) is specifically "declines correctly but pads
# with tangentially related results rather than stopping" - a clean decline
# in this agent's own traces runs to a sentence or two, so a generous word
# count catches "kept going" without trying to judge prose quality.
_DECLINE_PADDING_WORD_LIMIT = 80

# Wasted-cycle similarity threshold. difflib's SequenceMatcher ratio is a
# cheap, dependency-free proxy for "did the draft actually change" - it does
# not understand meaning, but restating the same facts in slightly different
# words (the failure mode this check exists for) still scores very high on
# it, which is exactly the case a stricter equality check would miss.
_WASTED_CYCLE_SIMILARITY_THRESHOLD = 0.95


def is_fixture_incomplete(record: RunRecord) -> bool:
    """True when a replay run aborted mid-turn on a FixtureMissError.

    Such a run is excluded from aggregation entirely: it stopped partway
    through on a missing fixture, so its tool counts and latency describe a
    truncated turn rather than the agent's behaviour.
    """
    return record.fixture_incomplete


@dataclass
class AssertionResult:
    """One check's outcome for one run.

    `known_defect` is set (by `evaluate_run`, not by the check_* function
    itself - a check has no business knowing about known_defects) exactly
    when `passed` is False and `question.known_defects` covers this key. The
    brief is explicit that this lookup is per assertion key, not per
    question: a question can have a known-broken citation check and a
    routing check that must still be able to break visibly.
    """

    key: str
    passed: bool
    detail: str = ""
    known_defect: str | None = None

    def __post_init__(self) -> None:
        # Guards against a check_* function drifting from the brief's fixed
        # vocabulary - if `key` cannot match a `known_defects` entry in
        # questions.yaml, a real known defect would silently read as a fresh
        # regression instead, which is precisely the failure mode ASSERTION_KEYS
        # exists to rule out (mirrors schema.py's own stance on TOOL_TO_ROUTE:
        # a naming mismatch must fail loudly, not masquerade as a genuine result).
        if self.key not in ASSERTION_KEYS:
            raise ValueError(f"Unrecognised assertion key {self.key!r}, expected one of {sorted(ASSERTION_KEYS)}")


@dataclass
class QuestionResult:
    """Every applicable assertion for one (question, run) pair."""

    question_id: str
    arm: str
    rep: int
    assertions: list[AssertionResult] = field(default_factory=list)

    @property
    def regressions(self) -> list[AssertionResult]:
        return [a for a in self.assertions if not a.passed and a.known_defect is None]

    @property
    def known_failures(self) -> list[AssertionResult]:
        return [a for a in self.assertions if not a.passed and a.known_defect is not None]


# --------------------------------------------------------------------------
# Individual checks. Each takes (question, record[, mode]) and returns None
# when the assertion does not apply - `evaluate_run` drops Nones rather than
# recording a vacuous pass, so a question with no must_contain entries does
# not silently inflate its own pass count.
# --------------------------------------------------------------------------


def check_routing(question: EvalQuestion, record: RunRecord) -> AssertionResult:
    """Did the run consult exactly `expected_routes`?

    Missing and unexpected are reported separately (not just "routing
    failed") because they are different defects: missing is under-research,
    unexpected is the redundancy-adjacent failure the web-question set exists
    to catch (a search_documents call on a question that sounds like it
    belongs to the corpus but does not).
    """
    expected = set(question.expected_routes)
    used = set(record.routes_used)
    missing = expected - used
    unexpected = used - expected
    passed = not missing and not unexpected
    parts = []
    if missing:
        parts.append(f"missing={sorted(r.value for r in missing)}")
    if unexpected:
        parts.append(f"unexpected={sorted(r.value for r in unexpected)}")
    return AssertionResult(
        key="routing", passed=passed, detail="; ".join(parts) or "routes matched exactly"
    )


def check_redundancy(question: EvalQuestion, record: RunRecord) -> AssertionResult:
    """Tool calls beyond `max_tool_calls`.

    Deliberately independent of check_routing: calling the right tool five
    times (the premise-refuting known defect) and calling the wrong tool once
    are different failures, and the brief is explicit they must be
    distinguishable rather than both just failing "routing".
    """
    n = len(record.tools_called)
    passed = n <= question.max_tool_calls
    return AssertionResult(
        key="redundancy",
        passed=passed,
        detail=f"{n} tool call(s) against a max of {question.max_tool_calls}: {record.tools_called}",
    )


def check_citation(question: EvalQuestion, record: RunRecord) -> AssertionResult | None:
    if not question.expects_citation:
        return None
    answer = record.answer

    # Strip the known-broken placeholder before looking for a real citation,
    # so an answer whose ONLY attribution is "(Source: Google Search)" cannot
    # satisfy the marker test on the word "Source" and pass.
    answer_without_placeholder = _BROKEN_CITATION_RE.sub(" ", answer)

    has_url = bool(_URL_RE.search(answer_without_placeholder))
    has_doc_reference = bool(_DOC_FILENAME_RE.search(answer_without_placeholder)) or (
        bool(_ATTRIBUTION_MARKER_RE.search(answer_without_placeholder))
        and bool(_DOCUMENT_NOUN_RE.search(answer_without_placeholder))
    )
    if has_url or has_doc_reference:
        kind = "url" if has_url else "document reference"
        return AssertionResult(key="citation", passed=True, detail=f"real citation present ({kind})")
    if _BROKEN_CITATION_RE.search(answer):
        return AssertionResult(
            key="citation",
            passed=False,
            detail="only the broken '(Google Search)' placeholder is present, no real URL or document reference",
        )
    return AssertionResult(key="citation", passed=False, detail="no citation of any kind found in the answer")


def check_decline(question: EvalQuestion, record: RunRecord) -> AssertionResult | None:
    if not question.expects_decline:
        return None
    answer_lower = record.answer.lower()
    declined = any(phrase in answer_lower for phrase in _DECLINE_PHRASES)
    if not declined:
        return AssertionResult(
            key="decline",
            passed=False,
            detail="no decline phrasing found - the agent may have fabricated an answer instead of declining",
        )
    word_count = len(record.answer.split())
    padded = word_count > _DECLINE_PADDING_WORD_LIMIT
    passed = not padded
    return AssertionResult(
        key="decline",
        passed=passed,
        detail=(
            f"declined plainly, {word_count} words (padding threshold {_DECLINE_PADDING_WORD_LIMIT})"
            if passed
            else f"declined but padded the answer to {word_count} words "
            f"(over the {_DECLINE_PADDING_WORD_LIMIT}-word threshold) with tangential content"
        ),
    )


def check_content(question: EvalQuestion, record: RunRecord) -> AssertionResult | None:
    """must_contain/must_not_contain, skipped for volatile questions outside replay.

    This split is the entire reason replay mode exists (references/
    evaluation_brainstorm.md): identical code and an identical question gave
    a different tool-call count and duration one day to the next purely
    because "yesterday" moved, so a volatile question's content can only be
    pinned against a recorded fixture, never against live results.
    """
    if not question.must_contain and not question.must_not_contain:
        return None
    if question.volatile and record.mode != "replay":
        return None
    missing = [s for s in question.must_contain if s not in record.answer]
    forbidden_present = [s for s in question.must_not_contain if s in record.answer]
    passed = not missing and not forbidden_present
    parts = []
    if missing:
        parts.append(f"missing={missing}")
    if forbidden_present:
        parts.append(f"forbidden_present={forbidden_present}")
    return AssertionResult(key="content", passed=passed, detail="; ".join(parts) or "all content expectations met")


def check_wasted_cycle(record: RunRecord) -> AssertionResult | None:
    """Did any cycle N+1 add no tool call while leaving the draft unchanged?

    Top-priority signal per the brief: a loop that burns budget restating the
    same answer had no detection at all before this tier. "No new tool call"
    alone is not sufficient - a cycle can legitimately add nothing new to
    search for and still improve wording/structure meaningfully - so both
    conditions (no new call AND draft barely moved) must hold together.
    Single-cycle runs have nothing to compare and are not applicable.
    """
    if record.cycle_count < 2:
        return None
    wasted: list[int] = []
    for prev, curr in zip(record.cycles, record.cycles[1:]):
        no_new_tool_call = len(curr.tools_called) == 0
        similarity = difflib.SequenceMatcher(None, prev.draft, curr.draft).ratio()
        if no_new_tool_call and similarity > _WASTED_CYCLE_SIMILARITY_THRESHOLD:
            wasted.append(curr.index)
    passed = not wasted
    detail = (
        "no wasted cycles"
        if passed
        else f"cycle(s) {wasted} added no tool call and left the draft materially "
        f"unchanged (similarity > {_WASTED_CYCLE_SIMILARITY_THRESHOLD}) - budget spent for no progress"
    )
    return AssertionResult(key="wasted_cycle", passed=passed, detail=detail)


def evaluate_run(question: EvalQuestion, record: RunRecord) -> QuestionResult:
    """Run every applicable check for one (question, run) pair.

    Caller's responsibility to have already excluded fixture-incomplete runs
    (see `is_fixture_incomplete`) - this function has no way to know that a
    run "tells you nothing" versus genuinely failing, since both look like a
    RunRecord with a short answer and few tool calls.
    """
    result = QuestionResult(question_id=record.question_id, arm=record.arm, rep=record.rep)
    checks = (
        check_routing(question, record),
        check_redundancy(question, record),
        check_citation(question, record),
        check_decline(question, record),
        check_content(question, record),
        check_wasted_cycle(record),
    )
    for res in checks:
        if res is None:
            continue
        if not res.passed and res.key in question.known_defects:
            res.known_defect = question.known_defects[res.key]
        result.assertions.append(res)
    return result


# --------------------------------------------------------------------------
# Latency: report-only, never gated (a 2x run-to-run spread would make a
# latency gate flap - references/evaluation_brainstorm.md), and always
# segmented by cycle count since the 15s target is only meaningful there.
# --------------------------------------------------------------------------


@dataclass
class LatencySegment:
    cycle_count: int
    n: int
    median_s: float
    iqr_s: float
    min_s: float
    max_s: float
    # None when this segment has no single-tool runs to judge against the
    # target (e.g. every run in a multi-cycle segment used >1 tool call) -
    # distinct from False, which would claim a real, failing measurement.
    meets_target: bool | None = None


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q1, _, q3 = statistics.quantiles(values, n=4)
    return q3 - q1


def summarise_latency(records: Iterable[RunRecord], contended: bool = False) -> dict[int, LatencySegment]:
    """Median/IQR latency segmented by cycle count, never a blended mean.

    Medians and spread only (never means): run-to-run spread on this stack is
    roughly 2x and the distribution has a long tail - two runaway traces once
    accounted for 86% of all output tokens the project had ever generated,
    which is exactly the kind of outlier a mean would let dominate silently.

    `contended` selects which population to summarise, and the two are never
    mixed: a run from run_sweep's concurrent phase shares the machine with
    others, so its wall-clock measures contention as much as the agent.
    Contended runs are segmented rather than dropped - `latency_target`
    questions are all single-cycle by design, so discarding the contended set
    would leave the multi-cycle segments (loop-multipart and friends) with no
    latency data at all. `meets_target` stays None for them regardless: the
    15s target was agreed over uncontended, single-tool turns only.
    """
    by_cycle: dict[int, list[RunRecord]] = {}
    for r in records:
        if r.timed_out or r.latency_s is None or r.contended != contended:
            continue
        by_cycle.setdefault(r.cycle_count, []).append(r)

    segments: dict[int, LatencySegment] = {}
    for cycle_count, recs in by_cycle.items():
        values = [r.latency_s for r in recs]
        meets_target: bool | None = None
        if cycle_count == 1 and not contended:
            single_tool_values = [r.latency_s for r in recs if len(r.tools_called) <= 1]
            if single_tool_values:
                meets_target = statistics.median(single_tool_values) <= LATENCY_TARGET_S
        segments[cycle_count] = LatencySegment(
            cycle_count=cycle_count,
            n=len(values),
            median_s=statistics.median(values),
            iqr_s=_iqr(values),
            min_s=min(values),
            max_s=max(values),
            meets_target=meets_target,
        )
    return segments


# --------------------------------------------------------------------------
# Per-arm rollup.
# --------------------------------------------------------------------------


@dataclass
class ArmSummary:
    arm: str
    n_runs: int
    n_fixture_incomplete: int
    n_timed_out: int
    n_errored: int
    # Runs that died on a Vertex quota rejection. Broken out of n_errored
    # because concurrency makes them expected rather than exceptional, and a
    # sweep whose "errors" are all quota rejections is a sweep to re-run at a
    # lower --concurrency, not a regression to investigate.
    n_rate_limited: int
    # Hard pass/fail per the brief: routing and redundancy only. Citation,
    # decline, content and wasted_cycle are still tracked (in regressions/
    # known_failures below) but do not feed this rate, since none of them are
    # asked to gate anything - only routing/redundancy are "hard pass/fail".
    hard_pass_rate: float
    regressions: list[tuple[str, AssertionResult]]
    known_failures: list[tuple[str, AssertionResult]]
    # Uncontended runs only - the population the 15s target is scoped to.
    latency_by_cycle: dict[int, LatencySegment]
    # Runs from run_sweep's concurrent phase. Kept in a separate field rather
    # than re-keying latency_by_cycle on (cycle_count, contended), which would
    # have rippled through every existing reader for no gain.
    latency_by_cycle_contended: dict[int, LatencySegment]
    # How many runs actually reached the assertions. Carried as a field rather
    # than derived by subtraction at print time: the excluded categories are
    # not disjoint (a rate-limited run also has `error` set), so subtracting
    # them from n_runs double-counts. That is exactly what it did before this
    # became a field.
    n_scored: int


_HARD_ASSERTION_KEYS = frozenset({"routing", "redundancy"})


def summarise(records: list[RunRecord], questions: list[EvalQuestion]) -> dict[str, ArmSummary]:
    """Roll a run set up per arm.

    Fixture-incomplete runs are excluded from every metric here (the brief:
    "excluded from metric aggregation and reported separately") - they are
    counted, not scored, since a run that aborted mid-turn tells you nothing
    about the code under test, only about fixture coverage. Timed-out runs
    are the opposite case: kept in the behavioural assertions (a timeout on a
    redundancy-bound question, e.g. the premise-refuting known defect, IS the
    defect manifesting) but counted separately and dropped from latency,
    where they have no finite value to report.
    """
    by_id = {q.id: q for q in questions}
    by_arm: dict[str, list[RunRecord]] = {}
    for r in records:
        by_arm.setdefault(r.arm, []).append(r)

    summaries: dict[str, ArmSummary] = {}
    for arm, arm_records in by_arm.items():
        incomplete = [r for r in arm_records if is_fixture_incomplete(r)]
        # A run that errored or timed out is excluded from assertion scoring
        # for the same reason a fixture-incomplete one is: it has no answer,
        # so every content-shaped assertion fails on it and the failures
        # describe the crash rather than the agent. Measured - three errored
        # runs in one 16-run sweep produced three phantom citation
        # regressions, which read exactly like a real attribution defect and
        # sent this investigation down the wrong path until the stored
        # answers were checked by hand.
        usable = [
            r
            for r in arm_records
            if not is_fixture_incomplete(r) and not r.timed_out and r.error is None
        ]
        timed_out = [r for r in arm_records if r.timed_out and not is_fixture_incomplete(r)]
        # A quota rejection is already excluded from `usable` (it sets
        # `error`), so this only splits the reporting: a rate-limited run says
        # something about how the sweep was scheduled, an errored one says
        # something about the agent, and conflating them would make a
        # too-high --concurrency look like a code regression.
        rate_limited = [r for r in arm_records if r.rate_limited]
        errored = [r for r in arm_records if r.error is not None and not r.timed_out and not r.rate_limited]

        regressions: list[tuple[str, AssertionResult]] = []
        known_failures: list[tuple[str, AssertionResult]] = []
        hard_total = 0
        hard_passed = 0

        for r in usable:
            question = by_id.get(r.question_id)
            if question is None:
                # A stored run referencing a question no longer in the
                # current set (e.g. questions.yaml was edited since the run
                # was recorded) - skip rather than crash the whole summary
                # over one stale row.
                continue
            qres = evaluate_run(question, r)
            for a in qres.assertions:
                if a.key in _HARD_ASSERTION_KEYS:
                    hard_total += 1
                    hard_passed += int(a.passed)
                if not a.passed:
                    (known_failures if a.known_defect else regressions).append((r.question_id, a))

        summaries[arm] = ArmSummary(
            arm=arm,
            n_runs=len(arm_records),
            n_fixture_incomplete=len(incomplete),
            n_timed_out=len(timed_out),
            n_errored=len(errored),
            n_rate_limited=len(rate_limited),
            hard_pass_rate=(hard_passed / hard_total) if hard_total else 1.0,
            regressions=regressions,
            known_failures=known_failures,
            latency_by_cycle=summarise_latency(usable, contended=False),
            latency_by_cycle_contended=summarise_latency(usable, contended=True),
            n_scored=len(usable),
        )
    return summaries


def print_summary(summaries: dict[str, ArmSummary]) -> None:
    """Human-readable rollup, unbuffered like run_eval.py's own progress output."""
    for arm, s in summaries.items():
        print(f"\n=== arm: {arm} ===", flush=True)
        print(
            f"  runs={s.n_runs} scored={s.n_scored} "
            f"fixture_incomplete={s.n_fixture_incomplete} timed_out={s.n_timed_out} "
            f"errored={s.n_errored} rate_limited={s.n_rate_limited} "
            f"hard_pass_rate={s.hard_pass_rate:.0%}",
            flush=True,
        )
        if s.regressions:
            print(f"  REGRESSIONS ({len(s.regressions)}):", flush=True)
            for qid, a in s.regressions:
                print(f"    - [{a.key}] {qid}: {a.detail}", flush=True)
        else:
            print("  no regressions", flush=True)
        if s.known_failures:
            print(f"  known defects still failing ({len(s.known_failures)}):", flush=True)
            for qid, a in s.known_failures:
                print(f"    - [{a.key}] {qid}: {a.detail} (ref: {a.known_defect})", flush=True)
        _print_latency(
            s.latency_by_cycle,
            "latency by cycle count, measured alone (median / IQR, seconds)",
        )
        # Printed under its own heading, never merged with the block above.
        # These runs shared the machine with up to --concurrency others, so
        # the numbers are indicative of throughput and say nothing about how
        # fast a turn is; presenting them in one table would invite exactly
        # that misreading.
        _print_latency(
            s.latency_by_cycle_contended,
            "latency by cycle count, measured under concurrency - NOT comparable "
            "with the above or with the 15s target",
        )


def _print_latency(segments: dict[int, LatencySegment], heading: str) -> None:
    if not segments:
        return
    print(f"  {heading}:", flush=True)
    for cycle_count in sorted(segments):
        seg = segments[cycle_count]
        # Label the SUBSET the verdict is actually about. The median printed
        # on this line covers every run in the cycle segment, but meets_target
        # is computed only over its single-tool runs, because that is the only
        # population the 15s target was agreed over. Without saying so, the
        # line reads "median=18.1s [OK <=15s]", which looks like a broken
        # comparison rather than two different populations.
        target = (
            ""
            if seg.meets_target is None
            else (
                " [single-tool subset OK <=15s]"
                if seg.meets_target
                else " [single-tool subset OVER 15s]"
            )
        )
        print(
            f"    cycles={cycle_count} n={seg.n} median={seg.median_s:.1f}s "
            f"iqr={seg.iqr_s:.1f}s min={seg.min_s:.1f}s max={seg.max_s:.1f}s{target}",
            flush=True,
        )
