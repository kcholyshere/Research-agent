"""Tier 1 evaluation runner - drives the question set through one or more
critique-budget arms and writes per-cycle RunRecords (see schema.py).

## How the per-cycle boundary was determined

Read src/research_agent/critique.py's module docstring first: `research_agent`
and `critique_agent` are the two sub-agents a LoopAgent cycles through, and
`critique_agent` has a `before_agent_callback` (`_skip_critique_llm_call`)
that can short-circuit its own LLM call entirely. That gave a hypothesis for
where cycle boundaries fall, but not their exact event shape - so it was
checked against real event streams (scripts under
/private/tmp/.../scratchpad/probe_events*.py, not kept in the repo) before
writing `_split_into_cycles` below. Three findings from that probing:

1. Every event in a turn carries `event.author` set to whichever sub-agent
   produced it (`research_agent`, `critique_agent`, or the LoopAgent itself,
   `research_loop`) - so author is a reliable phase marker with no need to
   infer it from content.
2. Both `LoopAgent.before_agent_callback` (`reset_turn_state`) and
   `critique_agent`'s own `before_agent_callback` mutate session state before
   their agent's real work happens. When they do, ADK emits a *separate*
   event carrying that state delta with `event.content is None`, ahead of
   whatever the agent's real model turn produces - confirmed by directly
   driving `critique_agent` with a synthetic incomplete draft, which reliably
   produced exactly two events: a `content is None` opener, then one real
   final-response event carrying the follow-up text. `research_agent` has no
   `before_agent_callback`, and never showed this opener event, which is
   consistent with the mechanism rather than coincidental. Filtering on
   `event.content is not None` (not just `is_final_response()`) is therefore
   required - an unfiltered final-response check treats the vacuous opener as
   real content and, for `critique_agent`, would close a cycle one event too
   early with an empty outcome.
3. The three critique outcomes are distinguishable without touching session
   state at all, from the event stream alone:
   - "skipped": `_skip_critique_llm_call` returns Content directly, so ADK
     emits exactly one event for the whole `critique_agent` block, with real
     text starting "Skipping critique:". No opener event precedes it, because
     the skip path never writes `critique_iterations_used` (see finding 2).
   - "exit": the block includes a `get_function_calls()`/`get_function_responses()`
     entry named "exit_loop"; the closing event has `escalate=True`,
     `skip_summarization=True` (per `exit_loop`'s own docstring) and text
     that is the tool's JSON return value, not a real follow-up.
   - "continue": no exit_loop call anywhere in the block; the closing event's
     text is the follow-up question(s) `critique_agent`'s output_key would
     write to `critique_followups`.

A live end-to-end run that actually continues past one cycle was not
observed in verification - this agent's critique prompt makes termination the
default outcome by design, and every live probe (including a purpose-built
multi-part comparison question) exited after one cycle. The "continue" event
shape above was confirmed by invoking `critique_agent` directly against a
deliberately incomplete synthetic draft, not by a full multi-cycle `root_agent`
turn - see this module's own report back to the user for that gap. The
splitting logic here does not special-case "budget 0" or "budget 1" - it is
driven entirely by event authorship and content, so it does not need to,
which is also part of why it survived not seeing a live multi-cycle run.

## Other traps this module has to honour (see references/evaluation_
brainstorm.md's "Traps found the hard way" for the measurements behind each)

- Fresh session per run, including repetitions - a reused session lets a
  later rep see an earlier rep's answer and short-circuit the work.
- Arms interleaved (A, B, A, B) across reps, mirroring ab_harness.compare's
  loop nesting (rep outer, arm middle, question inner) rather than running
  all reps of one arm before the next - that nesting is what shares
  live-world drift across arms instead of dumping it all on one.
- Sequential only - concurrent arms would contend for the same Vertex quota
  and inflate exactly the latency being measured.
- A per-run timeout, because nothing in this stack bounds a turn's duration
  on its own (one early harness run sat 85 minutes on a call that never
  returned).
- A FixtureMissError aborts a run, not the whole sweep: caught per run, the
  run is tagged fixture-incomplete (RunRecord.fixture_incomplete) and the
  loop moves on to the next run. That flag excludes it from aggregation
  entirely, since a turn truncated on a missing fixture describes the
  fixture rather than the agent.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import time
from pathlib import Path

import yaml
from google.adk.runners import InMemoryRunner
from google.genai import types

from src import config
from src.evaluation.langfuse_sync import link_run_to_dataset, push_question_set
from src.evaluation.metrics import print_summary, summarise
from src.evaluation.replay import FixtureMissError, FixtureMode, ToolFixtureSweep
from src.evaluation.schema import CycleRecord, EvalQuestion, EvalRun, RunRecord
from src.research_agent.agent import langfuse_client, research_agent, root_agent
from src.research_agent.critique import CRITIQUE_AGENT_NAME

APP_NAME = "eval_runner"
USER_ID = "eval-runner"

DEFAULT_QUESTION_FILE = config.PROJECT_ROOT / "data" / "eval" / "questions.yaml"
DEFAULT_FIXTURES_DIR = config.PROJECT_ROOT / "data" / "eval" / "fixtures"
DEFAULT_OUTPUT_DIR = config.PROJECT_ROOT / "data" / "processed" / "eval_runs"

# Stable across runs on purpose - a default that changed per invocation would
# scatter every sweep into its own Langfuse dataset instead of accumulating
# comparable arms in one place, defeating the whole point of the sync (see
# langfuse_sync.py's module docstring: "so arms are comparable in the
# existing UI").
DEFAULT_LANGFUSE_DATASET = "research-agent-tier1-eval"
DEFAULT_LANGFUSE_DESCRIPTION = (
    "Tier 1 trace-level evaluation question set (data/eval/questions.yaml) - "
    "see references/evaluation_brainstorm.md."
)

# Generous relative to a single-cycle turn (10-45s observed) because a
# critique budget above 0 can multiply that by the iteration count, and the
# premise-refuting known defect alone burns ~100s on its own. Still finite -
# see the module docstring's timeout trap for why this must never be None.
DEFAULT_TIMEOUT_S = 300.0

# Concurrent runs in the untimed phase (see run_sweep). 4 rather than higher
# because one turn is more than one Vertex request - research_agent,
# critique_agent and the web_search_agent sub-agent each call the model - so
# the real concurrent request count is a multiple of this, and quota
# rejections start costing more than the parallelism buys.
DEFAULT_CONCURRENCY = 4

# Substrings that identify a Vertex quota rejection in an exception's repr.
# Matched on text because the google-genai client surfaces these as a generic
# ClientError carrying the status in its message rather than as a distinct
# exception type - checked against a real rejection, not the docs.
_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")


def _looks_rate_limited(exc: BaseException) -> bool:
    text = f"{exc!r}".lower()
    return any(marker.lower() in text for marker in _RATE_LIMIT_MARKERS)


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


def load_questions(
    path: Path,
    question_ids: set[str] | None = None,
    tags: set[str] | None = None,
) -> list[EvalQuestion]:
    """Load and filter the question set.

    Filters are AND'd when both are given - `--questions` pins an exact set,
    `--tags` narrows by category; a caller combining both presumably means
    both, and it costs nothing to honour that literally.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [EvalQuestion.from_dict(q) for q in raw["questions"]]
    if question_ids is not None:
        questions = [q for q in questions if q.id in question_ids]
    if tags is not None:
        questions = [q for q in questions if tags.intersection(q.tags)]
    return questions


# ---------------------------------------------------------------------------
# Event stream -> per-cycle records
# ---------------------------------------------------------------------------


def _event_text(event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text or "" for part in event.content.parts if part.text)


def _split_into_cycles(events: list) -> list[CycleRecord]:
    """Reconstruct CycleRecords from a turn's raw event stream.

    See this module's docstring for how the boundary and the three critique
    outcomes ("exit" | "continue" | "skipped") were determined empirically.
    Deliberately tolerant of a stream that ends mid-cycle (a FixtureMissError
    or timeout can cut it off after research_agent but before critique_agent
    has run) - the partial cycle is still appended rather than dropped, since
    whatever tool calls and draft text were captured are real data the caller
    is told to keep.
    """
    cycles: list[CycleRecord] = []
    current: CycleRecord | None = None
    exit_seen = False

    for event in events:
        if event.author == research_agent.name:
            if current is None:
                current = CycleRecord(index=len(cycles))
                exit_seen = False
            current.tools_called.extend(call.name for call in event.get_function_calls())

            # Capture the phase 6 artefact from create_canvas's function
            # RESPONSE, not from the agent's prose. On an artefact turn the
            # answer is a short covering note by design (agent.py's step 4
            # says so), and the deliverable is the thing the assertions care
            # about - so an artefact read out of the answer text would be
            # whatever the model chose to paste, not what it actually
            # rendered.
            #
            # Verified against a real turn on google-adk 2.5.0 rather than
            # assumed: the function-response event for a tool called by
            # research_agent is itself authored by research_agent (so it lands
            # in this branch), and `response` is the tool's returned dict with
            # "artefact", "format" and "path" intact - a 4,287-character
            # artefact arrived whole, not truncated or stringified.
            #
            # Last write wins within a cycle. A cycle that rendered twice
            # after a validation error should record what it ended up with,
            # and cross-cycle re-renders stay visible because each cycle keeps
            # its own record (see CycleRecord.artefact).
            for response in event.get_function_responses():
                if response.name != "create_canvas":
                    continue
                payload = response.response
                if not isinstance(payload, dict) or payload.get("status") != "ok":
                    # An error return is a real outcome, not an artefact: the
                    # `artefact` assertion should see "nothing was produced"
                    # so a validation failure the agent never recovered from
                    # reads as a failure rather than as an absent field.
                    continue
                current.artefact = payload.get("artefact", "")
                current.artefact_format = payload.get("format", "")
                current.artefact_path = payload.get("path", "")

            if event.content is not None and event.is_final_response():
                text = _event_text(event)
                if text:
                    current.draft = text

        elif event.author == CRITIQUE_AGENT_NAME:
            if current is None:
                # Not seen in practice (critique_agent always follows a
                # research_agent cycle within a LoopAgent pass) but a run cut
                # short by a fixture miss or timeout is exactly the kind of
                # abnormal stream this should not crash on.
                current = CycleRecord(index=len(cycles))
                exit_seen = False

            called_or_returned = [c.name for c in event.get_function_calls()]
            called_or_returned += [r.name for r in event.get_function_responses()]
            if "exit_loop" in called_or_returned:
                exit_seen = True

            if event.content is not None and event.is_final_response():
                text = _event_text(event)
                if text.startswith("Skipping critique:"):
                    current.critique_outcome = "skipped"
                elif exit_seen:
                    current.critique_outcome = "exit"
                else:
                    current.critique_outcome = "continue"
                    current.critique_followups = text
                cycles.append(current)
                current = None

        # Any other author (today, only the LoopAgent's own before_agent_
        # callback bookkeeping event) carries no per-cycle signal and is
        # skipped - see finding 2 in the module docstring.

    if current is not None:
        cycles.append(current)
    return cycles


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


async def run_once(
    question: EvalQuestion,
    arm: str,
    budget: int,
    rep: int,
    mode: FixtureMode,
    fixture_sweep: ToolFixtureSweep,
    fixtures_dir: Path,
    timeout_s: float,
    contended: bool = False,
) -> RunRecord:
    """One question, once, against one arm, in its own fresh session.

    Mirrors ab_harness.run_once's shape (fresh session, author-matched answer
    extraction, wait_for timeout) but persists the full per-cycle structure
    rather than just latency and a flat tool list - RunRecord is a superset
    built for exactly that, per schema.py's own docstring.

    `contended` is recorded, not enforced: the caller decides whether this run
    shares the machine with others (run_sweep's concurrent phase), and only it
    can know. Recording it here keeps the flag with the latency it qualifies.

    `fixture_sweep` is the ONE `ToolFixtureSweep` for the whole sweep (its
    patches are already installed by the time any `run_once` call happens -
    see run_sweep) - this call only opens `fixture_sweep.run(...)`, the
    per-turn dispatch context, which is what makes it safe to call this
    concurrently for several questions at once (see replay.py's module
    docstring, "Concurrent sweeps").
    """
    record = RunRecord(
        question_id=question.id,
        question=question.question,
        arm=arm,
        rep=rep,
        mode=mode,
        contended=contended,
    )
    events: list = []
    fixture_path = fixtures_dir / f"{question.id}.json"

    async def _inner() -> None:
        runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        content = types.Content(role="user", parts=[types.Part(text=question.question)])
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=content,
            state_delta={"critique_budget": budget},
        ):
            events.append(event)
            # Cheap and non-blocking by construction (a contextvar read, no
            # network call) - but only obtainable while OTEL's span context
            # is still active, which is only true *during* iteration; a
            # measured probe found it reliably returns None once run_async's
            # generator has been fully exhausted. Captured once, opportunistically.
            if record.trace_id is None:
                try:
                    record.trace_id = langfuse_client.get_current_trace_id()
                except Exception:
                    pass  # never let trace-id capture affect a run's outcome

    started = time.monotonic()
    with fixture_sweep.run(fixture_path) as fixture_session:
        try:
            await asyncio.wait_for(_inner(), timeout=timeout_s)
            record.latency_s = time.monotonic() - started
        except asyncio.TimeoutError:
            record.timed_out = True
            record.latency_s = None
        except FixtureMissError as exc:
            # Abort this run only, not the sweep - keep whatever was captured
            # up to the miss (see _split_into_cycles's tolerance for a
            # mid-cycle cutoff) and mark it fixture-incomplete so it is
            # excluded from aggregation rather than scored as a failure.
            record.error = str(exc)
            record.fixture_incomplete = True
            record.latency_s = time.monotonic() - started
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill a 30-60 min sweep
            record.error = f"error: {exc!r}"
            record.latency_s = time.monotonic() - started
            # A quota rejection is not a result about the agent, so flag it for
            # exclusion rather than letting it score as a failure (see
            # RunRecord.rate_limited). Deliberately after `error` is set, not
            # instead of it - the message stays readable in the run file.
            record.rate_limited = _looks_rate_limited(exc)

        if mode != "live" and (fixture_session.misses or fixture_session.inexact_matches):
            print(
                f"    [fixtures] misses={len(fixture_session.misses)} "
                f"inexact_matches={len(fixture_session.inexact_matches)}",
                flush=True,
            )

    record.cycles = _split_into_cycles(events)

    # answer = the last non-empty final response authored by research_agent,
    # never "the last final response of any author" (that picks up whichever
    # sub-agent happened to speak last - the critique agent's "Skipping
    # critique: ..." bookkeeping or exit_loop's raw JSON) and never
    # session.state["draft_answer"] (can hold planning narration). This has
    # been got wrong three separate times in this project already.
    for event in events:
        if event.author == research_agent.name and event.content is not None and event.is_final_response():
            text = _event_text(event)
            if text:
                record.answer = text

    return record


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


LATENCY_TARGET_TAG = "latency_target"


def _plan_runs(
    questions: list[EvalQuestion], budgets: list[int], reps: int
) -> list[tuple[int, str, int, EvalQuestion]]:
    """Every (rep, arm, question) triple, in interleaved order.

    Loop nesting matches ab_harness.compare deliberately (rep outer, arm
    middle, question inner) - that is the nesting the interleaving trap was
    actually fixed against, so reproducing it here rather than inventing a
    different interleaving (e.g. question outer) keeps the same drift-sharing
    guarantee this project already paid to learn.
    """
    arms = {f"budget{b}": b for b in budgets}
    return [
        (rep, arm, budget, question)
        for rep in range(1, reps + 1)
        for arm, budget in arms.items()
        for question in questions
    ]


def _status(record: RunRecord) -> str:
    if record.rate_limited:
        return "RATE-LIMITED (not scored)"
    if record.timed_out:
        return "TIMEOUT"
    if record.error:
        return f"ERROR: {record.error[:80]}"
    return f"{record.latency_s:.1f}s cycles={record.cycle_count} tools={record.tools_called}"


async def run_sweep(
    questions: list[EvalQuestion],
    budgets: list[int],
    reps: int,
    mode: FixtureMode,
    fixtures_dir: Path,
    timeout_s: float,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[RunRecord]:
    """Every question through every arm, interleaved, `reps` times each.

    Run in two phases, because latency and throughput want opposite things.

    Phase 1 - questions tagged `latency_target` - is strictly sequential and
    is drained to completion before phase 2 starts. That ordering is the whole
    point: a timed run sharing the machine with a concurrent one is measuring
    contention rather than the agent, and the 15s single-tool target
    (metrics.LATENCY_TARGET_S) would become meaningless. Interleaving is
    preserved within the phase, since that is where the cross-arm latency
    comparison lives.

    Phase 2 - everything else - runs `concurrency` at a time. These questions
    are scored on routing, redundancy and content, none of which contention
    affects, so their wall-clock is the only casualty and it is recorded as
    `contended` rather than pretended away. Concurrency also subsumes what
    interleaving was doing for this phase: arms run literally simultaneously,
    so live-world drift cannot land on one arm rather than the other.

    Concurrency is no longer restricted by mode. `replay.ToolFixtureSweep`
    installs the tool patch exactly once for this whole sweep (below), and
    each `run_once` call dispatches through it via a `contextvars.ContextVar`
    scoped to that call's own `asyncio.Task` - concurrent turns resolve to
    their own fixture store and session with no shared mutable dispatch
    state (replay) or only record mode's deliberate, necessary sharing of one
    store per fixture path (see replay.py's module docstring, "Concurrent
    sweeps"). This replaces the mode != "live" => concurrency = 1 downgrade
    ADR-0012 recorded as deferred work.
    """
    plan = _plan_runs(questions, budgets, reps)
    total = len(plan)

    timed = [item for item in plan if LATENCY_TARGET_TAG in item[3].tags]
    untimed = [item for item in plan if LATENCY_TARGET_TAG not in item[3].tags]
    results: list[RunRecord] = []
    done = 0

    print(
        f"Phase 1: {len(timed)} timed run(s), sequential. "
        f"Phase 2: {len(untimed)} untimed run(s), {concurrency} at a time.",
        flush=True,
    )

    with ToolFixtureSweep(root_agent, mode) as fixture_sweep:
        for rep, arm, budget, question in timed:
            done += 1
            print(f"[{done}/{total}] timed rep={rep} arm={arm} question={question.id!r}...", end=" ", flush=True)
            record = await run_once(
                question, arm, budget, rep, mode, fixture_sweep, fixtures_dir, timeout_s, contended=False
            )
            results.append(record)
            print(_status(record), flush=True)

        if not untimed:
            return results

        semaphore = asyncio.Semaphore(concurrency)
        contended = concurrency > 1

        async def _guarded(item: tuple[int, str, int, EvalQuestion]) -> RunRecord:
            rep, arm, budget, question = item
            async with semaphore:
                record = await run_once(
                    question, arm, budget, rep, mode, fixture_sweep, fixtures_dir, timeout_s, contended=contended
                )
            nonlocal done
            done += 1
            # One complete line per completion, never the sequential phase's
            # "start ... finish" pair - concurrent runs would interleave the two
            # halves and produce unreadable output.
            print(
                f"[{done}/{total}] rep={rep} arm={arm} question={question.id!r}: {_status(record)}",
                flush=True,
            )
            return record

        # gather wraps each coroutine in its own Task, which is required for
        # two independent reasons: langfuse's get_current_trace_id reads
        # OTEL's active span from a contextvar, and replay.ToolFixtureSweep.run
        # sets _active_run on a contextvar too - both are copied per Task, so
        # driving these as bare coroutines on one Task would have them share a
        # context, attributing every run's trace id to whichever ran last and
        # letting concurrent turns see each other's fixture dispatch.
        results.extend(await asyncio.gather(*(_guarded(item) for item in untimed)))
    return results


# ---------------------------------------------------------------------------
# Langfuse sync (post-hoc, opt-in - never on the measurement path)
# ---------------------------------------------------------------------------


def sync_to_langfuse(eval_run: EvalRun, question_file: Path, dataset_name: str) -> None:
    """Mirror an already-finished `eval_run` into Langfuse. Never called mid-sweep.

    Both requirement sources here are `langfuse_sync.py`'s own module
    docstring ("a one-way, best-effort mirror of an already-finished run...
    never on the critical measurement path") and this project's repeated
    experience that coupling a measurement to an external service imports
    that service's failure modes into the number being measured - so this is
    only ever called after `eval_run` already exists (freshly written to disk
    by main_async, or loaded back from it via --sync-only). Both
    push_question_set and link_run_to_dataset already degrade to a logged
    warning on any failure and never raise (langfuse_sync._never_raises), so
    nothing here needs its own try/except to protect the caller's exit
    status or the JSON already on disk.

    Loads the FULL question file, not whichever subset this particular sweep
    was filtered to with --questions/--tags: dataset_name is a shared,
    long-lived Langfuse dataset that repeat runs land in (see
    DEFAULT_LANGFUSE_DATASET), and push_question_set is explicitly safe to
    call every time (upserts by id). Pushing only a smoke-test's one or two
    questions would leave the dataset an incomplete fragment of questions.yaml
    instead of a stable mirror of it; pushing the full file every time keeps
    it in sync regardless of what happened to be measured today. The same
    full set is passed to link_run_to_dataset purely for scoring labels
    (routing_correct/redundant_calls need EvalQuestion.expected_routes) - it
    does not change which records get linked, only whether their scores can
    be computed.
    """
    print(f"\nSyncing to Langfuse dataset {dataset_name!r}...", flush=True)
    questions = load_questions(question_file)
    push_question_set(questions, dataset_name, description=DEFAULT_LANGFUSE_DESCRIPTION)
    link_run_to_dataset(eval_run, dataset_name, questions=questions)
    print("Langfuse sync attempted (see warnings above for anything skipped).", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tier 1 trace-level evaluation runner.")
    parser.add_argument("--question-file", type=Path, default=DEFAULT_QUESTION_FILE)
    parser.add_argument(
        "--questions", type=str, default=None, help="Comma-separated question ids to run (smoke mode)."
    )
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated tags to filter by (smoke mode).")
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Untimed runs to execute at a time. latency_target questions always run sequentially "
        "and complete before any concurrent run starts. Applies in every mode, including "
        "record/replay - see replay.ToolFixtureSweep.",
    )
    parser.add_argument("--mode", choices=["live", "record", "replay"], default="live")
    parser.add_argument(
        "--budgets", type=str, default="0,1", help="Comma-separated critique budgets, one arm per value."
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, dest="timeout_s")
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-summary", action="store_true", help="Skip printing the metrics.py rollup after the sweep."
    )
    parser.add_argument(
        "--sync-langfuse",
        action="store_true",
        help=(
            "After the sweep has written its JSON, push the question set and link this run into "
            "Langfuse as a dataset run (post-hoc; a sync failure never affects the run's exit status)."
        ),
    )
    parser.add_argument(
        "--langfuse-dataset-name",
        type=str,
        default=DEFAULT_LANGFUSE_DATASET,
        help="Langfuse dataset name for --sync-langfuse/--sync-only; defaults to a stable, shared name "
        "so repeat runs land in the same dataset.",
    )
    parser.add_argument(
        "--sync-only",
        type=Path,
        default=None,
        metavar="PATH",
        help="Skip the sweep entirely: load an existing EvalRun JSON file from PATH and sync it to "
        "Langfuse (implies --sync-langfuse). Lets a sync be fixed or retried without re-running a sweep.",
    )
    return parser


async def main_async(argv: list[str] | None = None) -> EvalRun:
    args = build_arg_parser().parse_args(argv)

    if args.sync_only is not None:
        # A fix-the-sync-without-repeating-the-sweep escape hatch (see the
        # flag's own help text) - deliberately the only thing this branch
        # does: no question filtering, no sweep, no re-writing the JSON that
        # is already the source of truth on disk. EvalRun.from_json is the
        # same loader link_run_to_dataset's own docstring assumes callers use.
        eval_run = EvalRun.from_json(args.sync_only)
        sync_to_langfuse(eval_run, args.question_file, args.langfuse_dataset_name)
        return eval_run

    question_ids = _parse_str_set(args.questions)
    tags = _parse_str_set(args.tags)
    budgets = _parse_int_list(args.budgets)

    questions = load_questions(args.question_file, question_ids, tags)
    if not questions:
        raise SystemExit("No questions matched --questions/--tags filters - nothing to run.")

    print(
        f"Running {len(questions)} question(s) x {len(budgets)} arm(s) x {args.reps} rep(s) "
        f"= {len(questions) * len(budgets) * args.reps} runs, mode={args.mode}",
        flush=True,
    )

    started_at = dt.datetime.now(dt.timezone.utc)
    records = await run_sweep(
        questions=questions,
        budgets=budgets,
        reps=args.reps,
        mode=args.mode,
        fixtures_dir=args.fixtures_dir,
        timeout_s=args.timeout_s,
        concurrency=args.concurrency,
    )
    finished_at = dt.datetime.now(dt.timezone.utc)

    settings = {
        "question_file": str(args.question_file),
        "question_ids_filter": sorted(question_ids) if question_ids else None,
        "tags_filter": sorted(tags) if tags else None,
        "reps": args.reps,
        "mode": args.mode,
        "budgets": budgets,
        "timeout_s": args.timeout_s,
        # Recorded so a stored run is self-describing about how it was
        # measured - latency from a concurrency>1 sweep is only comparable
        # with another run's on the timed phase (see RunRecord.contended).
        "concurrency": args.concurrency,
        "fixtures_dir": str(args.fixtures_dir),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    eval_run = EvalRun(settings=settings, records=records)

    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    budgets_tag = "-".join(str(b) for b in budgets)
    output_path = args.output_dir / f"{timestamp}_{args.mode}_reps{args.reps}_budgets{budgets_tag}.json"
    eval_run.to_json(output_path)
    print(f"\nWrote {len(records)} run record(s) to {output_path}", flush=True)

    if not args.no_summary:
        summaries = summarise(records, questions)
        print_summary(summaries)

    if args.sync_langfuse:
        # Strictly after to_json above - the local JSON is already the
        # complete, correct record of this sweep by this point, so nothing
        # about the sync (success or failure) can change what got measured
        # or what got written.
        sync_to_langfuse(eval_run, args.question_file, args.langfuse_dataset_name)

    return eval_run


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
