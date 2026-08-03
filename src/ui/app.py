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
from pathlib import Path

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


async def _run_turn(
    runner: InMemoryRunner,
    session_id: str,
    message: str,
    critique_budget: int,
    web_search_thinking_budget: int,
) -> tuple[str, dict[str, str] | None]:
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
    final_text = "(no response)"
    # The phase 6 artefact, captured from create_canvas's function RESPONSE
    # rather than from the answer text. On an artefact turn the answer is a
    # short covering note by design (see step 4 of research_agent's
    # instruction), so a UI that rendered only the answer would show the user
    # "I have written the report to /path/..." and nothing else - the
    # deliverable would exist on disk and never appear on screen.
    artefact: dict[str, str] | None = None
    # session_id/user_id group this turn's spans into Langfuse's Sessions/Users
    # views - each chat_input submission is one ADK run, so one Langfuse trace.
    with propagate_attributes(session_id=session_id, user_id=USER_ID, tags=["research_agent"]):
        # critique_budget is the per-request soft cap the critique agent honours
        # (ADR-0010) - 0 short-circuits to a single research cycle with no
        # critique pass, matching pre-phase-4 behaviour.
        # web_search_thinking_budget is read the same way by web_search_agent's
        # before_model_callback (src/tools/web_search.py) - AgentTool copies
        # this session's state into the sub-agent's own session when
        # research_agent calls web_search_tool, so one state_delta here reaches
        # both agents' per-request knobs.
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content,
            state_delta={
                "critique_budget": critique_budget,
                "web_search_thinking_budget": web_search_thinking_budget,
            },
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
            if event.author == research_agent.name:
                # Same capture the evaluation harness does (run_eval._cycles),
                # for the same reason: the rendered artefact is in the tool
                # response, and only a successful render counts - an error
                # return is not a document.
                for response in event.get_function_responses():
                    if response.name != "create_canvas":
                        continue
                    payload = response.response
                    if isinstance(payload, dict) and payload.get("status") == "ok":
                        artefact = {
                            "text": payload.get("artefact", ""),
                            "format": payload.get("format", ""),
                            "language": payload.get("language", ""),
                            "path": payload.get("path", ""),
                        }
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
    return final_text, artefact


# Filename extension -> (Streamlit code language, download MIME type). The
# artefact's own format string drives both, so a fourth Canvas format needs one
# entry here rather than a new branch.
_ARTEFACT_MIME = {"markdown": "text/markdown", "html": "text/html", "code": "text/plain"}


def _render_artefact(artefact: dict[str, str], key_suffix: str) -> None:
    """Show a Canvas artefact below the answer, with a download.

    Rendered per format rather than uniformly, because the useful view differs:

    - markdown renders as markdown, which is what the document is meant to look
      like. "$" is escaped for the same reason the answer escapes it - financial
      artefacts are full of dollar figures, and st.markdown reads a pair of them
      as a LaTeX span and garbles everything between two unrelated amounts.
    - html is shown as SOURCE, not rendered. st.html would be the obvious
      choice and is the wrong one: Canvas emits a complete standalone document
      (doctype, head, its own <style>), and injecting that into a page that
      already has both would have the artefact's CSS leak into the app's own
      layout. The download plus "open the file" is the honest presentation of a
      standalone page.
    - code is shown as code, highlighted with the artefact's own language.
      Passing it matters more than it looks: st.code's `language` defaults to
      "python" (verified against the installed Streamlit), so omitting it does
      not mean "no highlighting" - it means every artefact is highlighted AS
      Python, and a SQL or JavaScript file is quietly mislabelled. `None` is the
      correct fallback for a language Canvas did not record, giving plain
      monospace rather than a confident wrong guess.
    """
    text = artefact.get("text", "")
    if not text:
        return
    fmt = artefact.get("format", "")
    path = artefact.get("path", "")
    name = Path(path).name if path else f"artefact.{fmt or 'txt'}"

    with st.container(border=True):
        st.caption(f":material/description: Artefact - {fmt or 'unknown format'} - `{name}`")
        if fmt == "markdown":
            st.markdown(text.replace("$", "\\$"))
        elif fmt == "html":
            st.code(text, language="html")
        else:
            st.code(text, language=artefact.get("language") or None)
        st.download_button(
            "Download artefact",
            data=text,
            file_name=name,
            mime=_ARTEFACT_MIME.get(fmt, "text/plain"),
            icon=":material/download:",
            # Keyed on position in the conversation, not the filename. Two
            # artefacts in one session can share a name - the same request asked
            # twice inside the same second produces the same title and the same
            # timestamp - and a duplicate widget key is a hard Streamlit error,
            # which would take the whole page down mid-demo rather than
            # degrading.
            key=f"download-{key_suffix}",
        )


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

# One expander for advanced, infrequently-touched per-request settings.
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

    # Named discrete levels rather than a raw token count: the number itself
    # (256 vs 512 vs 1024) means little to someone tuning latency, the level
    # does. select_slider returns the label; _web_search_thinking_levels maps
    # it back to the token budget web_search_agent's before_model_callback
    # applies (src/tools/web_search.py). Measured (n=6, direct Vertex probe):
    # unset ("automatic" thinking) put the web search sub-agent at a median
    # 19.95s/3,310 thinking tokens per call; 512 measured 9.46s for the same
    # probe - google_search grounding needs little deliberation, so most of
    # that thinking time was overhead the answer didn't need.
    _web_search_thinking_levels = {"Off": 0, "Low": 256, "Medium": 512, "High": 1024}
    # Falls back to "Medium" rather than raising if DEFAULT_WEB_SEARCH_THINKING_BUDGET
    # is ever set to a value outside these four levels (e.g. -1, Gemini's own
    # "automatic" budget - a legitimate config value, just not one of the
    # slider's named levels) - a wrong default slider position is recoverable
    # by the user; a crashed page on load is not.
    _default_thinking_level = next(
        (
            label
            for label, budget in _web_search_thinking_levels.items()
            if budget == config.DEFAULT_WEB_SEARCH_THINKING_BUDGET
        ),
        "Medium",
    )
    _web_search_thinking_label = st.select_slider(
        "Web search thinking budget",
        options=list(_web_search_thinking_levels),
        value=_default_thinking_level,
        help=(
            "How much the web search sub-agent is allowed to 'think' before "
            "answering. Off disables thinking entirely; higher levels let it "
            "reason more before replying, at the cost of extra latency per "
            "web search call. Google Search grounding needs little "
            "deliberation, so Medium is a reasonable default."
        ),
    )
    web_search_thinking_budget = _web_search_thinking_levels[_web_search_thinking_label]

runner = _runner()
session_id = asyncio.run(_ensure_session(runner))

if "history" not in st.session_state:
    st.session_state["history"] = []

for turn_index, turn in enumerate(st.session_state["history"]):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        # Replayed from history rather than rendered once: Streamlit reruns the
        # whole script on every interaction, so an artefact shown only in the
        # branch that produced it disappears the moment the user touches a
        # slider or sends another message.
        if turn.get("artefact"):
            _render_artefact(turn["artefact"], key_suffix=f"history-{turn_index}")

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
        turn_result: dict[str, object] = {}

        def _run_turn_sync() -> None:
            try:
                turn_result["answer"], turn_result["artefact"] = asyncio.run(
                    _run_turn(
                        runner,
                        session_id,
                        prompt,
                        critique_budget,
                        web_search_thinking_budget,
                    )
                )
            except Exception as exc:
                # Vertex errors, an empty/missing FAISS index, etc. should read as a
                # message in the chat, not crash the page.
                turn_result["answer"] = f"Error: {exc}"
                turn_result["artefact"] = None

        start_time = time.monotonic()
        turn_thread = threading.Thread(target=_run_turn_sync, daemon=True)
        turn_thread.start()
        with st.status("Thinking...", state="running") as status:
            while turn_thread.is_alive():
                status.update(label=f"Thinking for {time.monotonic() - start_time:.1f} seconds...")
                turn_thread.join(timeout=0.2)
            status.update(label=f"Thought for {time.monotonic() - start_time:.1f} seconds", state="complete")
        answer = str(turn_result["answer"]).replace("$", "\\$")
        st.markdown(answer)
        artefact = turn_result.get("artefact")
        if artefact:
            # "live" rather than a positional index: this render happens before
            # the turn is appended to history, so the index it would get here is
            # the one the history replay will also use on the next rerun - and
            # the same key appearing twice in one script run is the error this
            # avoids.
            _render_artefact(artefact, key_suffix="live")
    st.session_state["history"].append(
        {"role": "assistant", "content": answer, "artefact": artefact}
    )
