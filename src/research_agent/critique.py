"""Critique agent - the phase 4 key design task.

Phase 4 asks for a component that reviews the research agent's own answer,
finds gaps, and either lets a turn end or feeds follow-up questions back into
another research cycle. That is structurally the same impulse this project
has twice already had to suppress: Langfuse traces showed the root agent
re-searching a fact it had already answered because a result "surprised" it
or because it wanted to "confirm" something once more (commit 28f2c3d, and
the Document Search Tool's description, commit 390f940). A critique step with
an official mandate to "find gaps" can reintroduce exactly that behaviour
with the excuse now built into its job description. Most of this module's
design effort goes into preventing that, not into the critique logic itself.

Three decisions do the actual preventing (see agent_docs/decisions.md
ADR-0010 for the full reasoning):

1. Termination is the default outcome. `critique_agent`'s instruction
   requires it to name one specific, still-unanswered sub-question from the
   user's original query before it is allowed to continue the loop -
   anything else, including "let me double-check that", must call
   `exit_loop`. The model has to justify continuing; it never has to justify
   stopping. This is what makes "a fact already retrieved is not an
   unanswered sub-question" enforceable in wording rather than just hoped
   for.
2. `critique_agent` runs with `include_contents="none"`, so it never sees the
   research cycle's raw tool calls/passages - only the draft answer text
   (with its citations), injected via the `{draft_answer}` state variable
   that `research_agent`'s `output_key` writes. It cannot rediscover the urge
   to search because it never sees anything to search again *for* - an
   uncited claim or an unaddressed part of the question is all it has to
   work with.
3. The loop's stop condition is enforced by the orchestrator (ADK's
   `LoopAgent`, via `event.actions.escalate`), never requested of the model.
   `exit_loop` only ever sets that flag; it is not a judgement call the
   agent talks itself out of making.

On top of the model-level defence, one case is short-circuited in plain code
rather than left to the model at all: a single-fact financial lookup
(`get_financial_data` was the only tool the cycle used) has nothing to
critique, and a request-level `critique_budget` of 0 or already-spent must
skip the critique LLM call entirely, not just be told to approve quickly -
"the model chose to approve fast" is not the same guarantee as "no LLM call
happened here". Both checks live in a `before_agent_callback`, which is
deterministic Python, precisely because the whole point is that this
decision must not be a model judgement.

A separate correctness point, easy to miss: session state outlives a single
turn - `scripts/verify_agent.py` and the Streamlit UI both reuse one session
across many user turns - but `critique_iterations_used`, `original_query` and
`critique_followups` are only meaningful for the turn currently in flight.
Left alone, turn 2 of a multi-turn session would start already "spent" from
turn 1's critique call, would show critique_agent turn 1's question forever,
and could hand research_agent a stale follow-up question from a completely
unrelated earlier turn. `reset_turn_state`, wired in as the *loop's own*
before_agent_callback (not critique_agent's), runs exactly once per turn -
before either sub-agent's first cycle - and is what keeps these three keys
correctly turn-scoped despite living in otherwise session-scoped state.
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from google.genai import types

from src import config

CRITIQUE_AGENT_NAME = "critique_agent"


def exit_loop(tool_context: ToolContext) -> dict:
    """End the research loop for this turn.

    Call this when no unanswered sub-question remains: the draft answer
    covers every part of the user's original question and every claim in it
    is cited. Do not call anything else instead of this when you have
    decided the answer is complete - a plain text reply does not stop the
    loop, only this does.
    """
    tool_context.actions.escalate = True
    # ADK marks a "final response" event per participating agent, not just
    # once per turn (see Event.is_final_response's own docstring) - without
    # this, critique_agent would get one more model call to write closing
    # remarks, and that text (not research_agent's actual draft_answer)
    # would be liable to be picked up by anything that naively takes "the
    # last final-response event" as the answer to show. Suppressing it
    # removes both that ambiguity and the wasted extra LLM call - callers
    # should read session state's "draft_answer" for the real answer either
    # way, since that ambiguity exists whenever this returns False too.
    tool_context.actions.skip_summarization = True
    return {"status": "critique complete - ending the research loop"}


def _user_query_text(callback_context: CallbackContext) -> str:
    """Plain text of the message that started this turn.

    `critique_agent` runs with include_contents="none" (see the module
    docstring for why), so it has no conversation history of its own to read
    the original question from - it has to be handed over explicitly via
    session state instead (see reset_turn_state, below).
    """
    content = callback_context.user_content
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts)


def reset_turn_state(callback_context: CallbackContext) -> None:
    """Reset this turn's critique bookkeeping, once, before the loop starts.

    Wired as the LoopAgent's own before_agent_callback (see agent.py), not
    critique_agent's - that placement is what makes "once per turn" true
    without needing to compare invocation ids by hand: a LoopAgent's
    before_agent_callback fires exactly once per call to run_async, before
    its internal while-loop over sub_agents begins, and that internal loop is
    what re-runs research_agent/critique_agent on each refinement cycle.
    Doing this reset on critique_agent's own callback instead would be one
    cycle too late - research_agent's first cycle of a new turn would
    already have read whatever "critique_followups" was left over from the
    previous turn before critique_agent got a chance to clear it.
    """
    state = callback_context.state
    state["original_query"] = _user_query_text(callback_context)
    state["critique_iterations_used"] = 0
    state["critique_followups"] = ""


def _tools_used_this_cycle(callback_context: CallbackContext) -> set[str]:
    """Names of the tools called since the current research cycle began.

    Scans the session's event log backwards from the most recent event,
    stopping at the previous critique_agent turn or the user message that
    started this turn - whichever comes first. That boundary matters because
    `session.events` holds the whole conversation, not just this turn: with
    no stopping point, a long-running session (the Streamlit UI reuses one
    session across turns) could pull an earlier turn's tool calls into this
    turn's decision.
    """
    tool_names: set[str] = set()
    for event in reversed(callback_context.session.events):
        if event.author in (CRITIQUE_AGENT_NAME, "user"):
            break
        tool_names.update(call.name for call in event.get_function_calls())
    return tool_names


def _skip_critique_llm_call(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Deterministically decide whether critique_agent's LLM call happens.

    Returning non-None Content here (rather than just setting the escalate
    action and returning None) is load-bearing, not stylistic: ADK's
    before_agent_callback only skips the agent's own model call
    (`ctx.end_invocation = True`) when the callback returns truthy content -
    an escalate flag set on a None return still lets the LLM run this cycle
    before the loop notices it should have stopped. Returning content is what
    makes this an actual short-circuit rather than a slower way to reach the
    same LLM call.

    Two independent reasons to skip, checked every cycle:
    - The per-request budget (session state "critique_budget", defaulting to
      config.DEFAULT_CRITIQUE_BUDGET when the request didn't set one) is 0
      or has already been spent by an earlier cycle's real critique call.
      This is what makes a budget of 0 reproduce the pre-phase-4 single-cycle
      behaviour exactly: the very first check on the very first cycle finds
      0 >= 0 and escalates before anything resembling a critique runs.
    - The just-completed research cycle's only tool call was
      get_financial_data - a single live-price lookup has no citation to
      omit and no sub-question left to ask, so critiquing it is pure
      overhead on a path this project already measured close to its latency
      target.
    """
    # original_query/critique_iterations_used are already turn-fresh by the
    # time this runs - reset_turn_state (the loop's own before_agent_callback)
    # sets them before research_agent's first cycle of this turn.
    state = callback_context.state
    budget = state.get("critique_budget", config.DEFAULT_CRITIQUE_BUDGET)
    spent = state.get("critique_iterations_used", 0)
    budget_exhausted = spent >= budget

    financial_only = _tools_used_this_cycle(callback_context) == {"get_financial_data"}

    if budget_exhausted or financial_only:
        callback_context.actions.escalate = True
        reason = (
            "critique budget for this request is spent"
            if budget_exhausted
            else "cycle only looked up a live financial figure - nothing to critique"
        )
        return types.Content(
            role="model",
            parts=[types.Part(text=f"Skipping critique: {reason}.")],
        )

    # Proceeding to a real critique call - record it against the budget
    # before the call happens, not after, so a cycle that errors out
    # mid-call still counts as spent rather than being retried for free.
    state["critique_iterations_used"] = spent + 1
    return None


INSTRUCTION = """You review one research cycle's draft answer against the
user's original question. You do not see the search results or tool calls
that produced the draft - only the question and the draft itself - so you
cannot go and search anything yourself.

Original question:
{original_query}

Draft answer (with its citations):
{draft_answer}

Your only job is to decide whether a SPECIFIC part of the original question
above remains unanswered by the draft, or whether the draft states a claim
with no citation behind it. Termination is the default outcome: call
exit_loop immediately unless you can name one such specific gap. You must
justify continuing the loop; you never need to justify stopping it.

None of the following are grounds to continue - call exit_loop instead:
- A fact the draft already states, even if you would phrase it differently,
  want more precision, or want it said again more strongly.
- Wanting to re-verify, double-check, or confirm a fact the draft already
  answered. A fact already retrieved and stated is not an unanswered
  sub-question, no matter how confident you are that it deserves another
  look.
- A result that surprises you, or seems to contradict what you expected. A
  surprising answer is still an answer - it is evidence, not a defect, and
  is never grounds for another research cycle on its own.
- Wanting more detail, better phrasing, extra caveats, or context beyond
  what the original question actually asked for.

These are the only grounds to continue - if you find one, state it plainly
as a specific follow-up sub-question for the next research cycle, and do
not call exit_loop:
- A distinct part of the original question that the draft does not address
  at all.
- A factual claim in the draft with no source/citation attached to it.

If you found no such gap, call exit_loop now and say briefly why the answer
is complete. Do not call exit_loop and also propose a follow-up question -
pick one outcome.
"""

# Same runaway-repetition safety net as research_agent and web_search_agent
# (see ADR-0009) - this agent is a much shorter call in practice, but nothing
# about the failure mode is specific to answer length.
_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    max_output_tokens=4096,
    frequency_penalty=0.4,
)

critique_agent = Agent(
    name=CRITIQUE_AGENT_NAME,
    model=config.GEMINI_MODEL,
    description="Reviews a draft answer against the original question and either ends the research loop or raises specific follow-up questions.",
    instruction=INSTRUCTION,
    # include_contents="none": deliberately blind to the conversation/tool-
    # call history (see module docstring point 2) - the instruction's
    # {draft_answer}/{original_query} state injection is this agent's only
    # window onto the turn.
    include_contents="none",
    tools=[exit_loop],
    output_key="critique_followups",
    before_agent_callback=_skip_critique_llm_call,
    generate_content_config=_GENERATE_CONTENT_CONFIG,
)
