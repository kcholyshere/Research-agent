"""A cumulative token ceiling for one session, enforced in code.

This is the third bound in this project and the first one scoped to a
*session* rather than a single call or a single turn. The other two are
deliberately narrower and neither one substitutes for this:

- `max_output_tokens=4096` (ADR-0009) bounds one response. It was added after
  a decoding loop ran to the model's own 65,532-token ceiling.
- `MAX_TOOL_CALLS_PER_TURN` (tool_budget.py, ADR-0015) bounds one turn's
  evidence gathering.

Neither bounds what a user can spend across a conversation, and the cost of a
conversation is not the sum of its turns. Every turn resends the whole history,
so prompt tokens grow with the transcript: measured on this agent (2026-08-06,
two plain knowledge-base questions in one session), research_agent's prompt
went 3,548 -> 4,429 -> 6,627 within turn 1 and started turn 2 at 7,103. Turn 1
cost about 16.9k tokens in total, turn 2 about 18.5k, and that gap widens for
the rest of the session. A long chat is therefore quadratic in turns, and
nothing anywhere told it to stop.

## What this counts, and where

`usage_metadata.total_token_count` from every model response, summed into one
session-scoped counter. That figure is prompt + output + thinking as the
provider bills it, which is the thing being bounded - counting our own
characters would measure something adjacent to the cost rather than the cost.

Coverage is the point, so it is wired on all three agents that call a model:
`research_agent`, `critique_agent`, and the `web_search_agent` sub-agent
behind `web_search_tool`. Leaving the sub-agent out would leave the most
expensive single call in the system uncounted (ADR-0014 measured it at a
median 3,310 thinking tokens per call before its budget was pinned).

The sub-agent counts because `AgentTool.run_async` copies parent state into
the child session and forwards the child's state deltas back out
(`tool_context.state.update(event.actions.state_delta)`), verified against the
installed google-adk 2.5.0. Known imprecision, stated rather than hidden: that
is a copy-in/write-back, not an atomic add, so two web searches issued as
parallel function calls in one step would each start from the same base and
the later write would lose the earlier one's tokens. It undercounts in a rare
case; it never overcounts, and the ceiling is not a billing ledger.

## Why the state key is not reset per turn

`STATE_KEY` is deliberately absent from `critique.reset_turn_state`. That
callback is the LoopAgent's own `before_agent_callback` and is the one hook
here that fires exactly once per turn - which is precisely what makes it the
right home for `tool_budget.STATE_KEY` and the wrong home for this one. A
session counter cleared at the top of every turn is a per-turn counter that
can never reach a session ceiling, and it would fail silently: no error, no
warning, just a bound that never fires. Adding this key there is the specific
mistake to avoid, and reset_turn_state carries a comment saying so.

## What happens when the ceiling is reached

`enforce_session_token_budget` is a `before_model_callback`, so it runs before
the request is sent and short-circuits it: returning an `LlmResponse` from
that hook makes ADK skip the model call and hand the response back as if the
model had produced it (verified in `base_llm_flow._call_llm_async`). The
refusal is therefore free - a spent session costs nothing further, rather than
costing one last call to be told it is spent.

It also sets `actions.escalate`, which ends the critique loop for the turn.
Without that the LoopAgent would run research and critique round again up to
`MAX_CRITIQUE_ITERATIONS`, refusing each model call and emitting the same
notice three times.

The wording follows tool_budget.py's lesson: a returned value is read by the
model (and, here, shown to the user) as the response itself, not as an error,
so it has to say both that the budget is spent and what to do instead - start
a new session. A limit that does not name its own remedy gets one invented.

One consequence that looks like a bug and is not: research_agent has
`output_key="draft_answer"`, so the refusal becomes that turn's draft answer.
That is intended. It is what every caller reads (research_agent's last final
response, per CLAUDE.md's rule) and therefore what the user sees, which is the
point. The escalate above means critique_agent never runs to read it.
"""

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from src import config

# Session-state key holding the running total of tokens billed to this session.
# Session-scoped on purpose: NOT reset by critique.reset_turn_state - see the
# module docstring's "Why the state key is not reset per turn".
STATE_KEY = "session_tokens_used"


def _refusal(used: int) -> LlmResponse:
    """The response a session gets once its ceiling is reached.

    Plain user-facing prose, because this is handed back as the agent's own
    answer rather than as a tool error - the person reading it is the user,
    and their next action (open a new session) has to be in the text.
    """
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    text=(
                        "This conversation has reached its token limit "
                        f"({used:,} of {config.MAX_SESSION_TOKENS:,} used), so I can't "
                        "answer anything further in it. Each turn re-sends the whole "
                        "conversation, so a long chat costs progressively more per "
                        "question. Start a new conversation to continue - your last "
                        "question is worth repeating there, where it will be answered "
                        "against a fresh, and much cheaper, context."
                    )
                )
            ],
        ),
    )


def enforce_session_token_budget(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Refuse this model call if the session's token ceiling is already spent.

    Returning None lets the call proceed, which is the path every ordinary
    session takes. Returning an LlmResponse skips the model call entirely and
    is handed back in its place.

    `llm_request` is unused - the decision is about what the session has
    already spent, not about this request. Charging a request before it runs
    would need a token count for a prompt not yet sent, which is an extra
    round trip to `count_tokens` on every single call to make the ceiling
    trip one call earlier. The ceiling is a bound on total spend, so being
    one call late costs at most one call, and that call is itself bounded by
    max_output_tokens.
    """
    used = callback_context.state.get(STATE_KEY, 0)
    if used < config.MAX_SESSION_TOKENS:
        return None
    # Ends the critique loop as well as this call - see the module docstring.
    # Harmless where there is no loop to end (the web_search sub-agent), which
    # is why this is one shared callback rather than two near-identical ones.
    callback_context.actions.escalate = True
    return _refusal(used)


def accumulate_token_usage(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    """Add this response's billed tokens to the session total.

    Returning None always: this observes the response, it never rewrites it.

    Partial responses are skipped. Nothing in this repo's own entrypoints
    streams today (the Streamlit UI, the eval harness and verify_agent all use
    the default non-streaming mode), but `adk web` does, and in streaming mode
    the same usage figures arrive on intermediate chunks as well as the final
    response - counting both would inflate the total for one entrypoint and
    not the others.
    """
    if llm_response.partial:
        return None
    usage = llm_response.usage_metadata
    if usage is None or usage.total_token_count is None:
        return None
    used = callback_context.state.get(STATE_KEY, 0)
    callback_context.state[STATE_KEY] = used + usage.total_token_count
    return None
