"""Trim research_agent's resent history to the last N user turns.

The follow-up ADR-0021 named and deferred (agent_docs/decisions.md,
"Trim the history to the last N turns"). ADR-0021's own counter fixed the
*ceiling* on a session; this attacks the *growth* underneath it - every
model call research_agent makes resends the whole conversation so far, and
that transcript only ever gets longer. Measured there: research_agent's
prompt went 3,548 -> 4,429 -> 6,627 within turn 1 and started turn 2 at
7,103. This module keeps that number roughly flat past the first few turns
instead of letting it climb for the rest of the session.

This does NOT replace token_budget.py's ceiling, and isn't meant to: a
trimmed session is cheaper per turn, but a user can still hold it open
indefinitely (ADR-0021's own stated objection to this exact option). Both
bounds are needed - one caps the cost of each turn, the other caps how many
turns can run at all.

## Where the trim happens, and why not session.events

`before_model_callback` receives the same `LlmRequest` object that
`_call_llm_async` goes on to send to the model (verified against the
installed google-adk 2.5.0, `flows/llm_flows/base_llm_flow.py`:
`_call_llm_async` calls `_handle_before_model_callback(invocation_context,
llm_request, model_response_event)` and then keeps using that same
`llm_request` for the real call if the callback returns `None` - the exact
mechanism `token_budget.enforce_session_token_budget` already relies on to
short-circuit a call, and `web_search._apply_thinking_budget` already relies
on to mutate `llm_request.config` in place). Reassigning
`llm_request.contents` here is therefore honoured the same way: confirmed
empirically below, not just read from source - a trimmed request measurably
lowered `usage_metadata.prompt_token_count` on the very next call.

It is honoured for whichever agent's callback does the reassigning, not
globally - each `LlmAgent` builds and sends its own `LlmRequest`, so this is
wired on `research_agent` only. `critique_agent` runs with
`include_contents="none"` (critique.py) and never receives history in the
first place, and `web_search_agent` (src/tools/web_search.py) gets a brand
new `InMemorySessionService` and session per call (`AgentTool.run_async`,
verified against the installed package) - neither has anything to trim.
research_agent is the only agent in this system whose `llm_request.contents`
grows across turns at all.

The alternative - trimming `callback_context.session.events` instead of
`llm_request.contents` - was rejected. Events are the source of truth ADK
replays on every future call (this project's own multi-turn session, e.g.
Streamlit and scripts/verify_agent.py, is one long-lived `InMemoryRunner`
session across many user turns), and deleting from it would be permanent:
a later cycle needing something just discarded (a critique follow-up
question, `tool_budget`'s per-turn bookkeeping reading recent events - see
`critique._tools_used_this_cycle`) would silently lose it. Trimming
`llm_request.contents` instead only shortens what THIS ONE call sends; the
full history stays in the session for every other reader, and next call's
`llm_request.contents` is rebuilt fresh from that same full history and
trimmed again the same way. Cheaper to get wrong, too: a bug here costs one
call's context, not a permanently corrupted session.

## What a "turn" is here, and how a boundary is found

Printing `llm_request.contents` across a real multi-turn session (see
scripts/verify_history_trim.py) shows it is not one `Content` per user turn -
it is one per EVENT: the user's question, then a `Content` per tool call,
one per tool response, the final answer, and one more for whatever
critique_agent said (even a skip). A single plain question can be five or
more entries by the time a second turn starts. Naively keeping "the last N
contents" would as often as not cut a function_call away from its
function_response, which every downstream reader (starting with the actual
Gemini API call) treats as a malformed request, not a shorter one.

So a turn boundary is found structurally instead: a `Content` opens a new
user turn if, and only if, it has role "user", is not a tool response
(`function_response` part), and is not ADK's own cross-agent context wrapper.
That wrapper is what critique_agent's remarks show up as in research_agent's
history - ADK reformats another agent's reply as a "user" `Content` whose
first part is literally `Content(role="user",
parts=[Part(text="For context:"), Part(text="[agent] said: ...")])`
(verified against the installed package,
`flows/llm_flows/contents.py::_present_other_agent_message`) - so it is
excluded by the same check that excludes tool responses, using that fixed
first-part text as the marker. Everything between one such boundary and the
next - the question, every tool call/response, the final answer, and every
critique remark, across as many refinement cycles as that turn ran - is one
turn, trimmed as a single unit. This is also why "turn" here is the whole
user-turn, not a research/critique cycle: a mid-turn critique context entry
looks identical in shape to the one preceding the NEXT user question, and
only the boundary rule above (which a real question always satisfies and a
critique remark never does) tells them apart.

## What is never trimmed

The system instruction is not in `llm_request.contents` at all - it lives on
`llm_request.config.system_instruction`, a separate field this module never
touches.

The current, in-progress turn is always kept whole: the boundary list's last
entry marks where it starts, and the slice below never cuts into or past
that point, regardless of MAX_HISTORY_TURNS. Cutting into it would not be
trimming history - it would be deleting the very tool calls and responses
the turn in progress needs to finish itself, which is a correctness bug, not
a shorter prompt.

## The honest cost

This drops OLDER turns' own tool calls, search results and citations from
what research_agent sees - not just their final prose answers. A follow-up
that only needs the previous turn's stated conclusion ("what about the year
before?") is unaffected, because the retained window keeps whole turns, not
summaries of them (see MAX_HISTORY_TURNS in config.py for the window size
and the measurement behind it). A follow-up reaching further back than that
window, or asking to re-derive something from a source a dropped turn
already searched, gets a fresh, correct search rather than a wrong answer:
the tool is still there and the agent's own instruction already has it
search again if the retained context doesn't contain a fact it needs. What
is genuinely lost is the CHEAPNESS of answering from a much older turn's
already-fetched evidence, not correctness - and that is exactly the trade
ADR-0021 named when it deferred this as "silently drops context the agent
may need mid-conversation": it is no longer silent (this module explains
what dropped and why) and it is bounded (whole turns, never mid-turn), but a
question that reaches far enough back does cost a repeated search.
"""

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from src import config

# The first part ADK stamps onto its own cross-agent "context" wrapper
# (flows/llm_flows/contents.py::_present_other_agent_message) - fixed text,
# not something this project generates, and the one thing (besides a tool
# response) that keeps a role="user" Content from being a real turn boundary.
# See the module docstring's "What a 'turn' is here" for why this matters.
_CONTEXT_WRAPPER_TEXT = "For context:"


def _opens_a_turn(content: types.Content) -> bool:
    """True if `content` is a genuine user question, not a tool response or
    ADK's cross-agent context wrapper - see the module docstring."""
    if content.role != "user" or not content.parts:
        return False
    if any(part.function_response is not None for part in content.parts):
        return False
    first_text = content.parts[0].text
    if first_text is not None and first_text.strip() == _CONTEXT_WRAPPER_TEXT:
        return False
    return True


def trim_history(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Keep only the last MAX_HISTORY_TURNS complete turns, plus the turn in
    progress, in this request's contents.

    Returns None always - this only ever narrows `llm_request.contents` in
    place (by reassignment) and lets the call proceed; it never answers on
    the model's behalf, which is why (unlike token_budget's callback on this
    same agent) it has nothing to do with `actions.escalate`.

    `callback_context` is unused: the decision only needs the request's own
    contents, not anything from session state. Kept as a parameter because
    ADK invokes every before_model_callback with both by keyword (the same
    signature token_budget.enforce_session_token_budget and
    web_search._apply_thinking_budget already rely on).
    """
    contents = llm_request.contents
    turn_starts = [i for i, c in enumerate(contents) if _opens_a_turn(c)]

    # MAX_HISTORY_TURNS complete PAST turns, plus the turn in progress
    # (the last entry in turn_starts) - see the module docstring's "What is
    # never trimmed". Nothing to do if the session hasn't yet grown past
    # that window.
    keep_segments = config.MAX_HISTORY_TURNS + 1
    if len(turn_starts) <= keep_segments:
        return None

    keep_from = turn_starts[-keep_segments]
    llm_request.contents = contents[keep_from:]
    return None
