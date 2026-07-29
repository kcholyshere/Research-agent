"""Langfuse dataset-run sync for the Tier 1 evaluation pipeline - opt-in,
never on the critical measurement path.

references/evaluation_brainstorm.md (line 44) asks for eval runs to be
logged as Langfuse dataset runs "so arms are comparable in the existing UI".
That is the whole job of this module: it does not run questions, measure
anything, or decide pass/fail - src.evaluation.schema's EvalRun/RunRecord,
written by a runner that does not exist yet, is already the source of truth
on disk (data/processed/eval_runs/), and stays that way. This module is a
one-way, best-effort mirror of an already-finished run into Langfuse so the
comparison view there has something to show; nothing else in the pipeline
reads anything back from it.

Why "opt-in" is load-bearing, not a preference: the same evaluation plan
records that 18 of 47 traces returned empty observations on first fetch, and
that a hung Vertex/google_search call once stalled a harness for 85 minutes
with no visible progress. Coupling a measurement to Langfuse would import
both failure modes into the number being measured. So every public function
here degrades to a logged warning on any failure - dataset unreachable,
malformed record, whatever - and the two entry points never raise. Both
return None either way; a caller that ignores the return value gets exactly
what "strictly additive" implies: sync what it can, skip what it can't,
never slow or break the run it is trying to mirror.

## API surface, verified against the installed langfuse==4.14.1 (not docs -
CLAUDE.md is explicit that these have been wrong before), via
`inspect.signature`/`inspect.getsource` and one throwaway dataset
(`eval-harness-api-probe-5527dc95`, created against the real Langfuse
project during verification - see this change's report for cleanup notes):

- `Langfuse.create_dataset(name=...)` upserts by name - verified empirically
  by calling it twice with the same name and comparing the returned ids
  (identical both times). "Create the dataset if absent" therefore needs no
  existence check first; the obvious get-then-create race is moot because
  the create call is already safe to repeat.
- `Langfuse.create_dataset_item(dataset_name=, id=, ...)` upserts by `id`
  (stated in its own docstring, and confirmed here by creating the same `id`
  twice with different `input` and observing the second value win). Keying
  dataset items on `EvalQuestion.id` is what makes push_question_set safe to
  re-run as the question set evolves, without duplicating items.
- `client.api.dataset_run_items.create(run_name=, dataset_item_id=,
  trace_id=, ...)` is the actual item-to-run linkage primitive in this SDK
  version - there is no context-manager or `.run()` helper on `DatasetClient`
  for this. The closest thing, `DatasetClient.run_experiment`, drives its own
  task function over dataset items rather than accepting an already-finished
  external run, which is the wrong shape for "link a run that already
  happened, from a trace id we already have". `client.api.dataset_run_items`
  is the lower-level Fern-generated resource underneath the convenience
  methods and is fully public.
- That same probe showed `dataset_run_items.create` is NOT idempotent:
  calling it twice with identical `run_name`/`dataset_item_id`/`trace_id`
  produced two distinct run-item ids. Left unguarded, every re-run of
  `link_run_to_dataset` against an unchanged file would duplicate rows in
  Langfuse's run view.
- `get_dataset_run(...).dataset_run_items` returns only the most-recently-
  created item per `dataset_item_id`, NOT every row. This broke the first
  version of the guard, which compared (question_id, trace_id) pairs: on a
  multi-rep file only the last rep of each question looked already-linked, so
  a second sync re-linked and re-scored every earlier rep. Measured, not
  theorised - one trace reached 12 scores instead of 4 after three test
  syncs, and neither scores nor run items have a delete API here, so the
  duplicates were permanent. The guard is now whole-run (`_run_already_synced`):
  the default run name is a content fingerprint of the file, so "this run
  exists" already means "this file was synced".
- `get_dataset_run` has the same ingestion lag the plan already documents for
  trace/observation fetches (18 of 47 empty on first fetch): a run item
  created moments earlier in the same process was invisible to an immediate
  follow-up `get_dataset_run` call, then present a few seconds later with no
  code change - reproduced directly against the probe dataset, not inferred.
  In this project's actual use (a human re-syncing a file minutes or hours
  later) that lag is a non-issue; back-to-back calls seconds apart could
  still race past the guard. Documented rather than solved, since solving it
  would mean polling an eventually-consistent read with no documented bound.
- `create_score(trace_id=, name=, value=, data_type=)` is how a score
  attaches to a specific run's trace; `data_type="BOOLEAN"` takes 1.0/0.0
  (confirmed against langfuse/batch_evaluation.py's own BOOLEAN example -
  `value`'s declared type is `float | str`, not `bool`, so a bare Python
  `bool` is avoided here even though it happens to satisfy the isinstance
  checks). Scores are attached by `trace_id` alone, not `dataset_run_id` -
  the linked run-item is what associates a trace with a run in the
  comparison UI, so a trace-level score already shows up per-item there;
  `dataset_run_id` on `create_score` is for a different case (an aggregate
  score with no single trace behind it), not this one.
- `create_dataset`, `create_dataset_item` and `dataset_run_items.create` are
  synchronous REST calls (`self.api.<resource>.create(...)`, read straight
  from source - no queue involved), so no flush is needed for them.
  `create_score` is different: it calls `self._resources.add_score_task(...)`,
  a queued background event. Without an explicit `client.flush()` at the end
  of `link_run_to_dataset`, a short-lived script (this is one) can exit
  before the queue drains and the scores never leave the process.

## What was NOT verified live
No `delete_dataset` exists on either the convenience client or the raw
`DatasetsClient` (checked `dir()` on both) - the throwaway probe dataset
named above has no programmatic cleanup path and needs manual deletion in
the Langfuse UI, or just leaving it; it is clearly named as a probe.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from typing import TypeVar

from langfuse import Langfuse, get_client

from src import config  # noqa: F401 - triggers .env load via dotenv, see agent.py's import-order note
from src.evaluation.schema import EvalQuestion, EvalRun, RunRecord

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., None])


def _never_raises(fn: _F) -> _F:
    """Make a public entry point degrade to a logged warning, never raise.

    Both functions below already carry their own per-item try/except (one
    bad question or one bad record must not abort the rest of the sync) -
    this is the outer safety net for everything else: get_client() itself
    failing, a caller passing a broken iterable, a Langfuse SDK detail this
    file did not anticipate. "Strictly additive" (see module docstring)
    means this net has no gaps, not just that the common cases are handled.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("langfuse_sync.%s failed and was skipped: %s", fn.__name__, exc)

    return wrapper  # type: ignore[return-value]


@_never_raises
def push_question_set(
    questions: Iterable[EvalQuestion],
    dataset_name: str,
    *,
    description: str | None = None,
) -> None:
    """Push the Tier 1 question set to Langfuse as a dataset.

    Safe to call on every eval run, not just once: create_dataset upserts by
    name and create_dataset_item upserts by `id` (both verified - see module
    docstring), so re-running this after editing questions in place updates
    the existing dataset rather than duplicating it. A single bad question
    (unexpected content that fails to serialise, say) is logged and skipped
    rather than aborting the rest of the push - a 19-of-20 partial sync is
    still useful; one bad row aborting the whole set is not.
    """
    client = get_client()
    client.create_dataset(name=dataset_name, description=description)

    for question in questions:
        try:
            client.create_dataset_item(
                dataset_name=dataset_name,
                id=question.id,
                input={"question": question.question},
                # expected_output carries the actual pass/fail assertions -
                # schema.py's own framing for these fields is "assertions,
                # not documentation" - so a reviewer in the Langfuse UI sees
                # what a correct answer to this item was required to do, not
                # only what was asked.
                expected_output={
                    "expected_routes": [route.value for route in question.expected_routes],
                    "expects_citation": question.expects_citation,
                    "expects_decline": question.expects_decline,
                    "must_contain": question.must_contain,
                    "must_not_contain": question.must_not_contain,
                },
                metadata={
                    "tags": question.tags,
                    "max_tool_calls": question.max_tool_calls,
                    "volatile": question.volatile,
                    "known_defects": question.known_defects,
                    "notes": question.notes,
                },
            )
        except Exception as exc:
            logger.warning("langfuse_sync: dataset item %s failed and was skipped: %s", question.id, exc)
    # No flush here: create_dataset/create_dataset_item are synchronous REST
    # calls, not queued events (verified via source - see module docstring),
    # so there is nothing pending to flush by the time this returns.


def _content_fingerprint(eval_run: EvalRun) -> str:
    """Short, stable id for `eval_run`'s content - the default run-name stem.

    Nothing in this codebase constructs an EvalRun yet (grepped for it before
    writing this - no hits), so `settings`'s shape is not fixed enough to key
    a default name on a particular field (a "run_id" that may not exist, for
    instance). Hashing `settings` plus each record's (arm, rep, question_id)
    instead needs no agreed key, and - importantly - reproduces the same name
    for the same file across repeat calls, which is what lets the same
    `run_name` land on the same Langfuse run on a second sync; `_already_linked`
    (below) depends on that to recognise "this was already synced". It is not
    a strong uniqueness guarantee - two genuinely different runs with
    identical settings and record shape would collide - callers who need that
    guarantee should pass their own `run_name`.
    """
    fingerprint_input = {
        "settings": eval_run.settings,
        "records": [[r.arm, r.rep, r.question_id] for r in eval_run.records],
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"eval-{digest}"


def _run_already_synced(client: Langfuse, dataset_name: str, run_name: str) -> bool:
    """True when `run_name` already exists with at least one linked item.

    Deliberately a whole-run check rather than a per-record one. The first
    version compared (question_id, trace_id) pairs from
    `get_dataset_run(...).dataset_run_items` and dropped records it thought
    were already there - which was wrong, and measurably so: that endpoint
    returns only the most-recently-created run item per `dataset_item_id`,
    not every row. On a multi-rep file only the last rep of each question
    looked "already linked", so a second sync re-linked and re-scored every
    earlier rep. One trace ended up with 12 scores instead of 4 after three
    test syncs, and neither scores nor dataset run items have a delete API in
    this SDK version, so the duplicates could not be cleaned up.

    Because `run_name` defaults to a content fingerprint of the exact file
    (see _content_fingerprint), "this run exists" already means "this file
    was synced", so per-record bookkeeping buys nothing the name does not
    give. Trade-off accepted knowingly: a sync that failed halfway leaves a
    run that will now be skipped rather than completed - recover by passing
    an explicit `run_name`, which is cheap, whereas silently duplicating
    scores corrupts the data permanently.

    A run that does not exist raises inside get_dataset_run; that is the
    common first-sync case, not a failure, so it is swallowed here rather
    than logged. A genuine Langfuse outage still surfaces from the per-record
    create calls that follow.
    """
    try:
        run = client.get_dataset_run(dataset_name=dataset_name, run_name=run_name)
    except Exception:
        return False
    return bool(getattr(run, "dataset_run_items", None))


def _scores_for(record: RunRecord, question: EvalQuestion | None) -> list[tuple[str, float, str]]:
    """The four deterministic per-run scores, as (name, value, Langfuse data_type).

    All four come from RunRecord's own fields plus, where available, the
    EvalQuestion it was run against - none needs an LLM judge, which is the
    entire point of Tier 1 (references/evaluation_brainstorm.md: "every real
    bug so far... visible in trace structure alone, which is deterministic
    and costs nothing to assert on").

    routing_correct is EXACT-SET match between routes_used and
    expected_routes, not a subset check in either direction. The plan's own
    framing settles this: questions are "labelled with the tool routing each
    one should produce" (evaluation_brainstorm.md line 40) - "the routing it
    should produce" names one specific set, not a floor. schema.py's
    TOOL_TO_ROUTE docstring only states the over-routing half explicitly
    ("anything it consults beyond that set is a routing error"), but
    under-routing - never calling a source the question needed - is exactly
    as much a defect and has to fail the same assertion, or a run that
    silently skips a required source would score as correct.

    redundant_calls reuses `max_tool_calls`, a field schema.py already
    designed for exactly this ("Redundancy bound... calling the right tool
    five times is a different defect from calling the wrong tool once"),
    rather than inventing a second threshold here.

    Routing and redundancy are skipped (not scored 0/false) when `question`
    is None - a record whose question_id is not in the supplied question set
    (a stale run file against an edited dataset, say) has no label to check
    against, and a manufactured failure score would misrepresent that as the
    agent's fault rather than a caller/data mismatch. latency_s and
    cycle_count need no label and are always scored, timeouts included:
    latency_s is None on a timeout and is skipped since there is no number to
    report, but cycle_count (0 on a timeout with no completed cycle) is still
    real data worth recording.
    """
    scores: list[tuple[str, float, str]] = [("cycle_count", float(record.cycle_count), "NUMERIC")]

    if record.latency_s is not None:
        scores.append(("latency_s", record.latency_s, "NUMERIC"))

    if question is not None:
        routing_correct = set(record.routes_used) == set(question.expected_routes)
        scores.append(("routing_correct", 1.0 if routing_correct else 0.0, "BOOLEAN"))

        redundant_calls = max(0, len(record.tools_called) - question.max_tool_calls)
        scores.append(("redundant_calls", float(redundant_calls), "NUMERIC"))

    return scores


@_never_raises
def link_run_to_dataset(
    eval_run: EvalRun,
    dataset_name: str,
    *,
    run_name: str | None = None,
    questions: Iterable[EvalQuestion] = (),
) -> None:
    """Link a finished EvalRun's records into `dataset_name` as Langfuse dataset runs.

    Takes an already-loaded EvalRun (`EvalRun.from_json(path)`), never a path
    or a live measurement - the plan is explicit that this "operates on a
    finished run file, never inline during measurement", so there is no
    file-reading or run-executing code in this function at all, only sync.

    One Langfuse run PER ARM, not one run for the whole file: the plan's
    actual requirement (references/evaluation_brainstorm.md line 44) is that
    arms be comparable in Langfuse's run-comparison UI, which compares named
    runs against each other - a single run mixing every arm's records
    together would defeat that. `run_name` is the shared stem; the Langfuse
    run for arm "budget_0" is named "{run_name}-budget_0". If `run_name` is
    omitted, a stem is derived from the run's own content (see
    _content_fingerprint) so repeat calls on an unchanged file reuse the same
    Langfuse run rather than minting a new one every time.

    `questions` is optional: without it, every record still gets its
    latency_s/cycle_count scores, just not routing_correct/redundant_calls
    (see _scores_for for why those two specifically need the label).

    A record with no trace_id (a run that never reached Langfuse tracing, or
    timed out before one was captured) cannot be linked to anything and is
    skipped with a warning, the same as any other per-record failure - it
    does not abort the rest of the arm.
    """
    client = get_client()
    stem = run_name or _content_fingerprint(eval_run)
    question_index = {question.id: question for question in questions}

    for arm in dict.fromkeys(record.arm for record in eval_run.records):
        arm_run_name = f"{stem}-{arm}"
        if _run_already_synced(client, dataset_name, arm_run_name):
            logger.info(
                "langfuse_sync: run %s already exists in %s, skipping (pass an explicit "
                "run_name to force a fresh one)",
                arm_run_name,
                dataset_name,
            )
            continue

        for record in (r for r in eval_run.records if r.arm == arm):
            if not record.trace_id:
                logger.warning(
                    "langfuse_sync: record %s rep %s (arm %s) has no trace_id, skipping link",
                    record.question_id,
                    record.rep,
                    arm,
                )
                continue
            try:
                client.api.dataset_run_items.create(
                    run_name=arm_run_name,
                    dataset_item_id=record.question_id,
                    trace_id=record.trace_id,
                    metadata={"rep": record.rep, "mode": record.mode},
                )
                for name, value, data_type in _scores_for(record, question_index.get(record.question_id)):
                    client.create_score(
                        trace_id=record.trace_id,
                        name=name,
                        value=value,
                        data_type=data_type,
                    )
            except Exception as exc:
                logger.warning(
                    "langfuse_sync: linking record %s rep %s (arm %s) failed and was skipped: %s",
                    record.question_id,
                    record.rep,
                    arm,
                    exc,
                )

    # create_score is queued, not synchronous (verified via source - see
    # module docstring), so without this a short-lived script can exit before
    # the background queue drains and the scores never reach Langfuse at all.
    client.flush()
