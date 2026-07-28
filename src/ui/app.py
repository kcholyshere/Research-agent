"""Optional UI (phase 1 requirement: "Streamlit or Gradio for a simple
interface to the deployed agent"). A thin chat front end over root_agent -
the plan/execute/synthesize loop and the Document Search Tool itself live in
research_agent/agent.py and tools/document_search.py; this file only wires a
chat box to the ADK Runner.

Run with: uv run python -m streamlit run src/ui/app.py
(python -m, not the `streamlit` shim binary - see README's Setup section for why)
"""

import asyncio
import threading
import time

import streamlit as st
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from langfuse import get_client, propagate_attributes

from src import config
from src.research_agent.agent import research_agent, root_agent  # instruments ADK on import, see agent.py

APP_NAME = "research_agent"
USER_ID = "streamlit-user"

langfuse_client = get_client()


@st.cache_resource
def _runner() -> InMemoryRunner:
    return InMemoryRunner(agent=root_agent, app_name=APP_NAME)


async def _run_turn(runner: InMemoryRunner, session_id: str, message: str, critique_budget: int) -> str:
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
    final_text = "(no response)"
    # session_id/user_id group this turn's spans into Langfuse's Sessions/Users
    # views - each chat_input submission is one ADK run, so one Langfuse trace.
    with propagate_attributes(session_id=session_id, user_id=USER_ID, tags=["research_agent"]):
        # critique_budget is the per-request soft cap the critique agent honours
        # (ADR-0010) - 0 short-circuits to a single research cycle with no
        # critique pass, matching pre-phase-4 behaviour.
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
            state_delta={"critique_budget": critique_budget},
        ):
            # Match on the author, not just is_final_response(): since phase 4
            # (ADR-0010) the turn runs two agents, and ADK marks a final
            # response per participating agent rather than once per turn. The
            # critique agent's own last word is internal bookkeeping -
            # "Skipping critique: ..." or exit_loop's JSON status - so taking
            # the last final response of any author showed the user that
            # instead of the answer, on every single turn. Session state's
            # "draft_answer" is not a safe substitute either: it sometimes
            # holds the research agent's planning narration rather than its
            # answer. The research agent's own final response is the answer.
            if (
                event.author == research_agent.name
                and event.is_final_response()
                and event.content
                and event.content.parts
            ):
                final_text = "".join(part.text or "" for part in event.content.parts)
    # Streamlit is a long-lived server, not a short script, but flushing after
    # each turn keeps traces visible promptly rather than waiting on the SDK's
    # background batch export - worth the small per-turn cost here.
    langfuse_client.flush()
    return final_text


async def _ensure_session(runner: InMemoryRunner) -> str:
    if "session_id" not in st.session_state:
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        st.session_state["session_id"] = session.id
    return st.session_state["session_id"]


st.set_page_config(page_title="Research Agent", page_icon="\U0001f50e", layout="wide")
st.title("Research Agent")
st.caption(
    "Plan-execute-synthesize over a private knowledge base "
    f"(Document Search Tool, corpus: {config.RAW_DATA_DIR})"
)

# One expander for advanced, infrequently-touched per-request settings - a
# planned thinking-budget slider (ADR-0010) lands alongside critique_budget
# here later, so this stays a shared home for controls rather than a
# one-off widget bolted on next to the chat box.
with st.expander("Advanced configuration", expanded=False):
    critique_budget = st.slider(
        "Critique iterations",
        min_value=0,
        max_value=config.MAX_CRITIQUE_ITERATIONS,
        value=config.DEFAULT_CRITIQUE_BUDGET,
        help=(
            "How many times the agent critiques and refines its own answer "
            "before replying. 0 skips the critique pass entirely - fastest, "
            "and matches the agent's pre-phase-4 behaviour. Higher values let "
            "the agent spot gaps in its own answer and follow up, at the "
            "cost of extra latency per iteration."
        ),
    )

runner = _runner()
session_id = asyncio.run(_ensure_session(runner))

if "history" not in st.session_state:
    st.session_state["history"] = []

for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if prompt := st.chat_input("Ask a question about the knowledge base"):
    # Escape literal "$" - financial answers are full of dollar amounts, and
    # st.markdown treats a pair of "$" as a LaTeX math span, mangling anything
    # between two unrelated dollar figures into garbled italic notation.
    prompt = prompt.replace("$", "\\$")
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Run the turn on a background thread so the status label can keep
        # ticking up ("Thinking for x.x seconds...") while asyncio.run blocks.
        turn_result: dict[str, str] = {}

        def _run_turn_sync() -> None:
            try:
                turn_result["answer"] = asyncio.run(_run_turn(runner, session_id, prompt, critique_budget))
            except Exception as exc:
                # Vertex errors, an empty/missing FAISS index, etc. should read as a
                # message in the chat, not crash the page.
                turn_result["answer"] = f"Error: {exc}"

        start_time = time.monotonic()
        turn_thread = threading.Thread(target=_run_turn_sync, daemon=True)
        turn_thread.start()
        with st.status("Thinking...", state="running") as status:
            while turn_thread.is_alive():
                status.update(label=f"Thinking for {time.monotonic() - start_time:.1f} seconds...")
                turn_thread.join(timeout=0.2)
            status.update(label=f"Thought for {time.monotonic() - start_time:.1f} seconds", state="complete")
        answer = turn_result["answer"].replace("$", "\\$")
        st.markdown(answer)
    st.session_state["history"].append({"role": "assistant", "content": answer})
