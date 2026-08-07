"""Manual verification script for the phase 1 + phase 2 + phase 4 agent flow.

Deliberately still a script and not a test, now that `tests/` exists: an
LLM's exact tool choice is too non-deterministic to assert on reliably, so
there is no assertion this could make that would not eventually flake.
Measuring that behaviour is `src/evaluation/`'s job, over a question set and
a scored sweep; this is the human eyeball in between. Instead this drives the
same InMemoryRunner path app.py uses
over a small, deliberately varied set of questions, and prints which tool(s)
got called alongside each answer so a human can eyeball routing correctness.
Closes three open TODOS.md items in one pass: two-tool flow verification,
KB-lacks-answer decline behaviour, and confirming Langfuse traces actually
land (check the dashboard for session_id "verify-agent-smoke-test" after a
run).

KB-grounded questions are reused from Finrag's curated eval set
(../Finrag/references/RAG_evaluation_dataset.csv) - same source PDF as this
project's phase 1 corpus (ADR-0001), text/table-only to match this project's
ingestion (no image extraction, unlike Finrag's fuller pipeline).

Phase 4 (ADR-0010): `root_agent` is now a `LoopAgent` wrapping a research
agent and a critique agent, gated by a per-request `critique_budget` passed
via `runner.run_async(..., state_delta={"critique_budget": ...})`. One case
below is deliberately run twice - once with budget 0 (single research cycle,
no critique pass, the pre-phase-4 baseline) and once with the default budget
of 1 - printing elapsed time alongside each answer so the two runs can be
told apart: a budget-0 run should look identical to the old single-agent
flow, a budget-1 run should show a critique cycle's extra latency.

Run with: uv run python -m scripts.verify_agent
"""

import asyncio
import time

from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from langfuse import get_client, propagate_attributes

from src.config import DEFAULT_CRITIQUE_BUDGET
from src.research_agent.agent import research_agent, root_agent  # instruments ADK on import

APP_NAME = "research_agent"
USER_ID = "verify-agent-smoke-test"
SESSION_ID_TAG = "verify-agent-smoke-test"

langfuse_client = get_client()

# (label, question, expects) - "expects" is a human-readable note on what a
# correct run should look like, not something the script checks itself.
CASES = [
    (
        "kb-text",
        "What is the official title of the financial report for IFC's 2024 fiscal year?",
        "search_documents only; ground truth: 'IFC 2024 ANNUAL REPORT FINANCIALS'",
    ),
    (
        "kb-table",
        "What was IFC's Net Income for the fiscal year ending June 30, 2024?",
        "search_documents only; ground truth: $1,485 million",
    ),
    (
        "kb-table",
        "What was the total value of IFC's assets as of June 30, 2023?",
        "search_documents only; ground truth: $110,547 million",
    ),
    (
        "kb-text",
        "What is the stated mission of the International Finance Corporation?",
        "search_documents only; ground truth: end extreme poverty, boost shared prosperity",
    ),
    (
        "web-only",
        "Who is the current President of the World Bank Group?",
        "web_search_agent only - not in the FY2024 annual report",
    ),
    (
        "combined",
        "What was IFC's FY2024 net income, and what is IFC's current long-term "
        "issuer credit rating from a major ratings agency today?",
        "both tools - first half from KB ($1,485 million), second half needs live web search",
    ),
    (
        "should-decline",
        "What was IFC's average employee tenure in fiscal year 2024, broken "
        "down by employee's home country?",
        "neither source has this; agent should say so, not guess",
    ),
]

# Exercised separately from CASES above with two different critique_budget
# values, on the same question, so the two runs are otherwise comparable
# (see the phase 4 note in the module docstring).
CRITIQUE_BUDGET_QUESTION = "What was IFC's Net Income for the fiscal year ending June 30, 2024?"
CRITIQUE_BUDGETS_TO_COMPARE = [0, DEFAULT_CRITIQUE_BUDGET]


def _runner() -> InMemoryRunner:
    return InMemoryRunner(agent=root_agent, app_name=APP_NAME)


async def _run_case(
    runner: InMemoryRunner,
    session_id: str,
    label: str,
    question: str,
    expects: str,
    critique_budget: int = 0,
) -> None:
    print(f"\n{'=' * 80}\n[{label}] {question}\nexpected: {expects}\n{'-' * 80}")
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
    final_text = "(no response)"
    tools_called: list[str] = []
    start_time = time.monotonic()
    with propagate_attributes(session_id=session_id, user_id=USER_ID, tags=["verify_agent_smoke_test", label]):
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
            state_delta={"critique_budget": critique_budget},
        ):
            tools_called.extend(call.name for call in event.get_function_calls())
            # Author-matched for the same reason as the Streamlit UI: since
            # phase 4 the turn runs two agents and ADK marks a final response
            # per agent, so the critique agent's bookkeeping text would
            # otherwise be reported as the answer.
            if (
                event.author == research_agent.name
                and event.is_final_response()
                and event.content
                and event.content.parts
            ):
                final_text = "".join(part.text or "" for part in event.content.parts)
    elapsed = time.monotonic() - start_time
    langfuse_client.flush()
    print(f"critique_budget: {critique_budget}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"tools called: {tools_called or '(none)'}")
    print(f"answer: {final_text}")


async def _run_critique_budget_comparison(runner: InMemoryRunner) -> None:
    """Compare budgets on one question, each in its own fresh session.

    A fresh session per run is the whole point, not tidiness. Conversation
    history accumulates within a session, and Langfuse traces showed it
    inflating a turn's input from 7.5k to 14.8k tokens across one sitting.
    Reusing a session here would let the second run see the first run's
    answer already sitting in history, so it could short-circuit the work
    entirely - the elapsed-time difference would then measure the cache-like
    effect of history, not the cost of a critique cycle.
    """
    print(f"\n{'=' * 80}\n[critique-budget comparison] {CRITIQUE_BUDGET_QUESTION}")
    print("expected: same question, one fresh session each - compare elapsed time and answer")
    for critique_budget in CRITIQUE_BUDGETS_TO_COMPARE:
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        await _run_case(
            runner,
            session.id,
            f"critique-budget-{critique_budget}",
            CRITIQUE_BUDGET_QUESTION,
            "budget 0 mirrors the pre-phase-4 single-cycle flow (no critique pass); "
            f"budget {DEFAULT_CRITIQUE_BUDGET} (the default) may add a critique cycle and extra latency",
            critique_budget=critique_budget,
        )


async def main() -> None:
    runner = _runner()
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    for label, question, expects in CASES:
        await _run_case(runner, session.id, label, question, expects)
    await _run_critique_budget_comparison(runner)
    print(f"\n{'=' * 80}\nDone. Check the Langfuse dashboard for session_id={session.id!r} to confirm traces landed.")


if __name__ == "__main__":
    asyncio.run(main())
