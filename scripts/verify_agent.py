"""Manual verification script for the phase 1 + phase 2 agent flow.

Not a pytest suite: an LLM's exact tool choice is too non-deterministic to
assert on reliably, and this project has no tests/ directory yet (see
CLAUDE.md - the capstone's focus is agent architecture, not RAG/eval
engineering). Instead this drives the same InMemoryRunner path app.py uses
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

Run with: uv run python -m scripts.verify_agent
"""

import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from langfuse import get_client, propagate_attributes

from src.research_agent.agent import root_agent  # instruments ADK on import

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


def _runner() -> InMemoryRunner:
    return InMemoryRunner(agent=root_agent, app_name=APP_NAME)


async def _run_case(runner: InMemoryRunner, session_id: str, label: str, question: str, expects: str) -> None:
    print(f"\n{'=' * 80}\n[{label}] {question}\nexpected: {expects}\n{'-' * 80}")
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
    final_text = "(no response)"
    tools_called: list[str] = []
    with propagate_attributes(session_id=session_id, user_id=USER_ID, tags=["verify_agent_smoke_test", label]):
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
            tools_called.extend(call.name for call in event.get_function_calls())
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(part.text or "" for part in event.content.parts)
    langfuse_client.flush()
    print(f"tools called: {tools_called or '(none)'}")
    print(f"answer: {final_text}")


async def main() -> None:
    runner = _runner()
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    for label, question, expects in CASES:
        await _run_case(runner, session.id, label, question, expects)
    print(f"\n{'=' * 80}\nDone. Check the Langfuse dashboard for session_id={session.id!r} to confirm traces landed.")


if __name__ == "__main__":
    asyncio.run(main())
