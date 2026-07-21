"""Optional UI (phase 1 requirement: "Streamlit or Gradio for a simple
interface to the deployed agent"). A thin chat front end over root_agent -
the plan/execute/synthesize loop and the Document Search Tool itself live in
research_agent/agent.py and tools/document_search.py; this file only wires a
chat box to the ADK Runner.

Run with: uv run python -m streamlit run src/ui/app.py
(python -m, not the `streamlit` shim binary - see README's Setup section for why)
"""

import asyncio

import streamlit as st
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from src import config
from src.research_agent.agent import root_agent

APP_NAME = "research_agent"
USER_ID = "streamlit-user"


@st.cache_resource
def _runner() -> InMemoryRunner:
    return InMemoryRunner(agent=root_agent, app_name=APP_NAME)


async def _run_turn(runner: InMemoryRunner, session_id: str, message: str) -> str:
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
    final_text = "(no response)"
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts)
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

runner = _runner()
session_id = asyncio.run(_ensure_session(runner))

if "history" not in st.session_state:
    st.session_state["history"] = []

for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if prompt := st.chat_input("Ask a question about the knowledge base"):
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Planning and searching..."):
            try:
                answer = asyncio.run(_run_turn(runner, session_id, prompt))
            except Exception as exc:
                # Vertex errors, an empty/missing FAISS index, etc. should read as a
                # message in the chat, not crash the page.
                answer = f"Error: {exc}"
        st.markdown(answer)
    st.session_state["history"].append({"role": "assistant", "content": answer})
