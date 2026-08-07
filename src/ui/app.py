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
from src.research_agent import tool_budget
from src.research_agent.agent import research_agent, root_agent  # instruments ADK on import, see agent.py
from src.research_agent.critique import CRITIQUE_AGENT_NAME

APP_NAME = "research_agent"
USER_ID = "streamlit-user"

# Tool name -> what the user sees while it runs. Keyed by the name that appears
# in a real event's `call.name`, which for an AgentTool is the wrapped agent's
# name rather than the Python variable - "web_search_agent", not
# "web_search_tool". That distinction is the same one src/evaluation/schema.py's
# TOOL_TO_ROUTE comment exists to warn about, and getting it wrong here would
# show raw tool names in the UI rather than fail loudly.
#
# declare_plan and report_gap (agent_docs/audit.md finding 12) were added
# 2026-08-06 alongside the other five tools but never added here, so both
# rendered as their raw Python name in grey (the _STEP_LABELS.get/_STEP_COLOURS.get
# fallbacks below exist precisely so a missing entry degrades instead of
# crashing, which is also why the gap went unnoticed). tests/test_ui_refusal_display.py
# now checks coverage against research_agent's actual registered tool list
# rather than pinning this dict's keys by hand, so a sixth tool added the same
# way fails that test instead of waiting for a second audit pass.
_STEP_LABELS = {
    "declare_plan": "Declaring the research plan",
    "search_documents": "Searching the knowledge base",
    "web_search_agent": "Searching the web",
    "get_financial_data": "Fetching market data",
    "news_agent": "Talking to the News Agent",
    "create_canvas": "Building the artefact",
    "report_gap": "Recording a coverage gap",
}

# One colour per source, so a glance at the expander shows which mix of sources
# a turn used without reading the labels. Streamlit's markdown supports a fixed
# set of colour names in `:colour[text]`; these are all from that set.
#
# declare_plan and report_gap share create_canvas's "primary" colour rather
# than getting one each - deliberately, not for lack of a free colour name
# (only "yellow" is unused). All three are exactly tool_budget.OUTPUT_TOOLS:
# the tools that gather no evidence and are exempt from the numeric ceiling.
# Grouping them under one colour keeps the visual language "one colour per
# EVIDENCE source" true rather than diluting it with three more swatches for
# tools that never contribute a fact - a glance at the expander should still
# answer "which sources did this turn use", and now also "did it plan and
# record its outcome", without the two questions competing for colours.
_CRITIQUE_KIND = "critique"
_REFUSED_KIND = "refused"
_STEP_COLOURS = {
    "declare_plan": "primary",
    "search_documents": "blue",
    "web_search_agent": "green",
    "get_financial_data": "orange",
    "news_agent": "violet",
    "create_canvas": "primary",
    "report_gap": "primary",
    _CRITIQUE_KIND: "gray",
    _REFUSED_KIND: "red",
}

# exit_loop is the critique agent's loop-termination signal, not research work -
# it is bookkeeping the user has no use for, and it always takes ~0.0s.
_HIDDEN_STEPS = frozenset({"exit_loop"})


def _is_budget_refusal(payload: object) -> bool:
    """Whether a tool response is one of tool_budget.enforce_tool_budget's own refusals.

    agent_docs/audit.md finding 12: the check here used to compare
    `payload.get("error")` against the single literal
    "tool_call_budget_exhausted" - `_refusal`'s own string, and only that
    one. Two more refusal builders had been added the same day, each with its
    own "error" value this never learned, so a report_gap- or plan-refused
    call rendered as an ordinary, successful, instant tool call. Naming the
    missing strings would have reset the same trap for whoever adds a fifth.

    So this matches a marker key that every refusal in `tool_budget.py`
    carries and nothing else does, rather than any of their values. It picks
    up a new refusal builder for free, and it cannot collide with a tool's
    own domain error: `get_financial_data` and `search_documents` both return
    an "error" of their own, and `declare_plan` and `create_canvas` both
    return `{"status": "error", "detail": ...}` on a malformed call - which
    is byte-for-byte what `_amendment_refusal` returns too. No payload shape
    can separate those; only a marker can, which is why the marker exists.
    """
    return isinstance(payload, dict) and payload.get(tool_budget.REFUSAL_MARKER_KEY) == tool_budget.REFUSAL_MARKER


def _close_steps(steps: list[dict]) -> None:
    """Stop the clock on anything still running when a turn ends.

    Called from the UI thread's `finally` rather than from inside the turn,
    because the usual reason a step never closes is the turn RAISING - an
    unreachable News Agent surfaces as an ADK-level error rather than as a tool
    response (see src/tools/news_agent.py), so cleanup written after the event
    loop would be skipped exactly when it is needed. Every step, tool or
    critique phase, is in this one list, so a single scan closes them all.

    Without it the expander reads "Talking to the News Agent..." forever on a
    failed turn instead of giving the duration it actually spent trying.
    """
    now = time.monotonic()
    for step in steps:
        if step["ended"] is None:
            step["ended"] = now


def _duration(seconds: float) -> str:
    """Seconds at one decimal place, without a pointless trailing ".0".

    `:g` drops it: 6.0 renders "6s", 1.6 renders "1.6s". At this precision a
    ".0" is noise - it implies a resolution the measurement does not have.
    """
    return f"{round(seconds, 1):g}s"


def _format_steps(steps: list[dict]) -> str:
    """One line per step, for the body of the status expander.

    Deliberately returns a single markdown string rather than writing several
    elements: the live path fills a placeholder with this and the history replay
    renders it directly, so both produce exactly one markdown element inside the
    status. Element counts matching between the two paths is what stops the
    previous answer being stranded on screen - see the history loop below.

    Lines are joined with a markdown hard break rather than made a list, because
    a "-" bullet reads as a dash next to the coloured marker that now carries
    the same "this is an item" meaning.
    """
    lines = []
    for step in steps:
        colour = _STEP_COLOURS.get(step.get("kind"), "gray")
        marker = f":{colour}[■]"
        if step.get("refused"):
            # No duration: the call was blocked before it ran, so any number
            # here would be the cost of being refused, not of doing work.
            lines.append(f"{marker} {step['label']}")
        elif step["ended"] is None:
            lines.append(f"{marker} {step['label']}...")
        else:
            lines.append(
                f"{marker} {step['label']} for {_duration(step['ended'] - step['started'])}"
            )
    return "  \n".join(lines)

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
    steps: list[dict],
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
    # Live progress for the status expander. `steps` is owned by the UI thread
    # and appended to here: list.append and dict item assignment are atomic
    # under the GIL, and the reader only ever formats a snapshot, so no lock is
    # needed for the reader to stay consistent. Open tool calls are tracked by
    # the function call's `id` rather than its name, because the same tool is
    # called several times in a turn (the budget allows five) and matching a
    # response to the wrong call would attribute the wrong duration.
    open_calls: dict[str, dict] = {}
    critique_step: dict | None = None


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
            # The critique agent is not reached through a tool call, so it has
            # no call/response pair to time. Its phase is bounded by authorship
            # instead: it opens on the first event the critique agent emits and
            # closes when any other author speaks again (a second research
            # cycle) or when the turn ends.
            if event.author == CRITIQUE_AGENT_NAME:
                if critique_step is None:
                    critique_step = {
                        "label": "Critiquing the draft",
                        "kind": _CRITIQUE_KIND,
                        "started": time.monotonic(),
                        "ended": None,
                    }
                    steps.append(critique_step)
            elif critique_step is not None:
                critique_step["ended"] = time.monotonic()
                critique_step = None

            for call in event.get_function_calls():
                if call.name in _HIDDEN_STEPS:
                    continue
                step = {
                    "label": _STEP_LABELS.get(call.name, call.name),
                    "kind": call.name,
                    "started": time.monotonic(),
                    "ended": None,
                }
                open_calls[call.id] = step
                steps.append(step)
            for response in event.get_function_responses():
                step = open_calls.pop(response.id, None)
                if step is None:
                    continue
                step["ended"] = time.monotonic()
                # A call the tool budget refused never ran, so reporting it as
                # "Searching the web for 0s" is actively misleading - it reads
                # as a search that found nothing rather than one that was
                # blocked. Relabelling it says what actually happened, and is
                # the most useful line in the expander when a turn goes wide:
                # it marks the exact point the agent stopped being allowed to
                # gather more. Detected by response SHAPE via
                # _is_budget_refusal, not by a zero duration (which would also
                # match a genuinely instant call) and not by any one refusal's
                # "error" string (see that function's docstring for why, and
                # for the one refusal shape it cannot safely catch).
                payload = response.response
                if _is_budget_refusal(payload):
                    step["label"] = f"{step['label']}: refused by the tool-call gate"
                    step["kind"] = _REFUSED_KIND
                    step["refused"] = True

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
# entry here rather than a new branch. Same staleness shape as
# _STEP_LABELS/_STEP_COLOURS above (a fixed copy of an enum that lives in
# another module - canvas.OutputFormat), but with a softer failure: `.get(fmt,
# "text/plain")` below means a forgotten fourth format downloads with the
# wrong MIME type rather than crashing or rendering as raw internal text, so
# it is easy to miss in a demo. tests/test_ui_refusal_display.py checks this
# dict's keys against canvas.OutputFormat's actual Literal args for the same
# reason it checks _STEP_LABELS/_STEP_COLOURS against the registered tool
# list, rather than fixing the fallback here - a wrong-but-present MIME type
# is a real, if minor, defect worth a failing test, not a silent default.
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
        # Replayed for the same reason the artefact is, plus one specific to
        # Streamlit's rendering model. Streamlit diffs the element tree
        # positionally between runs, so a container whose children differ in
        # COUNT between the live render and the history render cannot be
        # matched up - the previous run's elements are left stranded on screen,
        # dimmed, until the new run happens to overwrite that position.
        #
        # That is exactly what a live-only status caused: the live assistant
        # block was [status, markdown], the history replay of the same turn was
        # [markdown], and the leftover showed as a greyed-out duplicate of the
        # last answer for as long as the next turn took to think. Rendering the
        # status here too keeps both paths structurally identical, and has the
        # side benefit that the timing stays visible instead of vanishing the
        # moment the user sends anything else.
        if turn.get("elapsed") is not None:
            with st.status(f"Thought for {_duration(turn['elapsed'])}", state="complete"):
                # Unconditional, even when the turn used no tools: the live path
                # always writes one markdown element in here, so this one must
                # too or the element counts diverge again.
                st.markdown(_format_steps(turn.get("steps") or []))
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
        # Run the turn on a background thread so the elapsed counter and the
        # step list can keep updating while asyncio.run blocks.
        turn_result: dict[str, object] = {}
        # Written by the turn thread as events arrive, read by the loop below to
        # redraw the expander - which is what makes the steps appear live rather
        # than all at once when the turn finishes.
        steps: list[dict] = []

        def _run_turn_sync() -> None:
            try:
                # config.TURN_TIMEOUT_S (agent_docs/TODOS.md): the whole-turn
                # bound. research_agent's before_agent_callback
                # (turn_deadline.enforce_turn_deadline) also stamps and checks
                # this same budget so adk run/adk web get a coarse version of
                # it too, but only this wait_for can actually cut off a single
                # slow cycle already in progress - the callback can only
                # refuse to START a new one. Wrapping asyncio.run's own
                # coroutine, not run_async's loop from the outside, so
                # cancellation reaches every await point inside _run_turn
                # (the model call, a tool call) rather than just the
                # boundary between events.
                turn_result["answer"], turn_result["artefact"] = asyncio.run(
                    asyncio.wait_for(
                        _run_turn(
                            runner,
                            session_id,
                            prompt,
                            critique_budget,
                            web_search_thinking_budget,
                            steps,
                        ),
                        timeout=config.TURN_TIMEOUT_S,
                    )
                )
            except TimeoutError:
                # Report plainly rather than surfacing a partial answer: a
                # cancelled turn is cut off mid-await, possibly between
                # research_agent finishing a cycle and critique_agent
                # reviewing it - exactly the half-written state CLAUDE.md's
                # answer-reading rule exists to protect against. There is no
                # event here that safely stands in for "the answer", so
                # this says what actually happened instead of guessing.
                turn_result["answer"] = (
                    f"This turn exceeded its {config.TURN_TIMEOUT_S:g}s time budget and "
                    "was stopped. No answer was produced - please try again, or split "
                    "the question into smaller parts."
                )
                turn_result["artefact"] = None
            except Exception as exc:
                # Vertex errors, an empty/missing FAISS index, etc. should read as a
                # message in the chat, not crash the page.
                turn_result["answer"] = f"Error: {exc}"
                turn_result["artefact"] = None
            finally:
                _close_steps(steps)

        start_time = time.monotonic()
        turn_thread = threading.Thread(target=_run_turn_sync, daemon=True)
        turn_thread.start()
        # The container is written ONCE, on creation, and not touched again
        # until the turn is over. That is the fix for the expander folding
        # itself shut the moment it was clicked: expanding is frontend-only
        # state, and re-sending the block every 0.2s - which a ticking label
        # requires - disturbs it. Nothing here can win that race, so the race is
        # removed instead of tuned.
        #
        # The cost is that the elapsed counter cannot live in the label while
        # running, because the label is part of the block. It moves into the
        # body, which is a child element and can be rewritten freely.
        #
        # Collapsed by default: this is a detail view, not the turn's output,
        # and it should not cost vertical space until asked for. Expanding it is
        # the user's decision, and because nothing rewrites the block, that
        # decision is never overridden - not while the turn runs, and not when
        # it finishes.
        with st.status("Thinking...", state="running") as status:
            # One placeholder, rewritten each tick, rather than a fresh element
            # per tick - otherwise every 0.2s poll would append another copy of
            # the step list to the expander.
            steps_slot = st.empty()
            while turn_thread.is_alive():
                ticker = f"**Thinking for {_duration(time.monotonic() - start_time)}**"
                steps_slot.markdown(f"{ticker}  \n{_format_steps(steps)}")
                turn_thread.join(timeout=0.2)
            # Captured rather than recomputed inside the label, because it is
            # stored on the turn and replayed by the history loop above - the
            # two renders have to agree on the number or the label would drift
            # by whatever the append costs.
            elapsed = time.monotonic() - start_time
            # Final redraw: the last few steps close after the loop's last tick,
            # so without this the expander would keep a step reading "..." even
            # though the turn is done.
            steps_slot.markdown(_format_steps(steps))
            # No `expanded` argument, deliberately. update() calls ClearField on
            # that field when it is None, which means "leave it as the user left
            # it" - so someone who opened the expander to watch the steps still
            # has it open when the answer lands, instead of having it snap shut
            # under them. It starts collapsed anyway, so there is nothing to
            # tidy away here.
            status.update(label=f"Thought for {_duration(elapsed)}", state="complete")
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
        {
            "role": "assistant",
            "content": answer,
            "artefact": artefact,
            "elapsed": elapsed,
            "steps": steps,
        }
    )
