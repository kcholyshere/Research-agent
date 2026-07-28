"""A/B comparison harness for agent variants - the seed of the evaluation pipeline.

Built to answer one question (had phase 4's new instruction step degraded the
research agent's search discipline?) but kept because three of its design
choices are the ones any latency or behaviour comparison here needs, and each
was learned by getting it wrong first:

1. Arms are INTERLEAVED (A, B, A, B), not run in blocks. The first version of
   this ran arm A twice and then arm B twice, which put each arm in its own
   time window - so drift in live web results or Vertex load would land
   entirely on one arm and read as a real effect. Interleaving splits that
   drift across both arms instead.
2. Every run gets a FRESH SESSION. Conversation history accumulates within an
   ADK session, and Langfuse traces showed it inflating a turn's input from
   7.5k to 14.8k tokens across one sitting. Reusing a session lets a later run
   see an earlier run's answer and short-circuit the work, so the measured
   difference becomes an artefact of history rather than of the variable
   under test.
3. Every run has a TIMEOUT. Nothing in this stack bounds a turn's duration:
   an early version of this harness sat 85 minutes on a Vertex/google_search
   call that never returned, having burned 6 seconds of CPU. Without a
   per-run bound, one hung call stalls an entire comparison silently.

What it deliberately does not do: judge answer quality. This measures latency
and tool-call behaviour only - the things assertable without a model in the
loop. Answer-quality scoring is a separate, later tier (see the evaluation
plan); keeping the two apart is what makes this cheap enough to run often.

Latency here is wall-clock per run, and run-to-run spread on this stack is
roughly 2x, so treat a single pair of runs as indicative only and prefer the
median of several repetitions.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field

from google.adk.agents import BaseAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

APP_NAME = "ab_harness"
USER_ID = "ab-harness"

# A hung model or grounding call must not stall the whole comparison. Generous
# relative to a normal turn (10-45s observed) so it only fires on a real hang.
DEFAULT_RUN_TIMEOUT_S = 240


@dataclass
class RunResult:
    """One question, run once, against one arm."""

    arm: str
    question: str
    latency_s: float | None  # None when the run timed out
    tools_called: list[str] = field(default_factory=list)
    answer: str = ""
    timed_out: bool = False

    def tool_count(self, tool_name: str) -> int:
        return sum(1 for name in self.tools_called if name == tool_name)


async def run_once(
    agent: BaseAgent,
    question: str,
    arm: str = "",
    state_delta: dict | None = None,
    timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    answer_author: str | None = None,
) -> RunResult:
    """Run one question through one agent in its own fresh session.

    `answer_author` names the agent whose final response counts as the answer,
    defaulting to `agent` itself. It has to be settable because a wrapper
    agent never authors the answer: pass the `root_agent` LoopAgent and the
    text comes from its `research_agent` sub-agent, so defaulting to the
    top-level name silently yields an empty answer. Latency and tool calls
    are unaffected either way - only the captured text depends on this.
    """
    expected_author = answer_author or agent.name

    async def _inner() -> RunResult:
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        content = types.Content(role="user", parts=[types.Part(text=question)])
        tools_called: list[str] = []
        answer = ""
        started = time.monotonic()
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=content,
            state_delta=state_delta,
        ):
            tools_called.extend(call.name for call in event.get_function_calls())
            # Author-matched deliberately: a multi-agent turn emits a final
            # response per participating agent, so an unfiltered "last final
            # response" picks up whichever agent happened to speak last rather
            # than the one that produced the answer.
            if (
                event.author == expected_author
                and event.is_final_response()
                and event.content
                and event.content.parts
            ):
                answer = "".join(part.text or "" for part in event.content.parts)
        return RunResult(
            arm=arm,
            question=question,
            latency_s=time.monotonic() - started,
            tools_called=tools_called,
            answer=answer,
        )

    try:
        return await asyncio.wait_for(_inner(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return RunResult(arm=arm, question=question, latency_s=None, timed_out=True)


async def compare(
    arms: dict[str, BaseAgent],
    questions: list[str],
    reps: int = 3,
    state_delta: dict | None = None,
    timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    answer_author: str | None = None,
    verbose: bool = True,
) -> list[RunResult]:
    """Run every question through every arm, interleaved, `reps` times each.

    Sequential on purpose. Running arms concurrently would have them contend
    for the same Vertex quota, which inflates exactly the latency being
    measured - so a comparison of N arms costs N times the wall clock, and
    that cost is the reason to keep question sets small.
    """
    results: list[RunResult] = []
    for rep in range(1, reps + 1):
        for arm_name, agent in arms.items():
            for question in questions:
                result = await run_once(
                    agent,
                    question,
                    arm=arm_name,
                    state_delta=state_delta,
                    timeout_s=timeout_s,
                    answer_author=answer_author,
                )
                results.append(result)
                if verbose:
                    if result.timed_out:
                        print(f"  [{arm_name}] rep{rep}: TIMEOUT after {timeout_s}s | {question[:50]!r}")
                    else:
                        print(
                            f"  [{arm_name}] rep{rep}: {result.latency_s:6.1f}s "
                            f"tools={result.tools_called} | {question[:50]!r}"
                        )
    if verbose:
        summarise(results)
    return results


def summarise(results: list[RunResult]) -> None:
    """Print median latency and tool-call counts per arm.

    Median rather than mean: this stack's latency distribution is skewed with
    a long tail (a runaway generation loop once produced a 4m35s turn), and a
    single outlier drags a mean somewhere no run actually went.
    """
    print(f"\n{'arm':32} {'n':>3} {'median':>9} {'min':>8} {'max':>8}  tool calls/run")
    for arm_name in dict.fromkeys(r.arm for r in results):
        arm_runs = [r for r in results if r.arm == arm_name]
        timings = [r.latency_s for r in arm_runs if r.latency_s is not None]
        timeouts = sum(1 for r in arm_runs if r.timed_out)
        counts = [len(r.tools_called) for r in arm_runs if not r.timed_out]
        if not timings:
            print(f"{arm_name:32} {len(arm_runs):>3}  all runs timed out")
            continue
        suffix = f" ({timeouts} timed out)" if timeouts else ""
        print(
            f"{arm_name:32} {len(timings):>3} {statistics.median(timings):8.1f}s "
            f"{min(timings):7.1f}s {max(timings):7.1f}s  {counts}{suffix}"
        )
