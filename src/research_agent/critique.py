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

On top of the model-level defence, three cases are short-circuited in plain
code rather than left to the model at all: a single-fact financial lookup
(`get_financial_data` was the only evidence tool the cycle used) has nothing
to critique, a request-level `critique_budget` of 0 or already-spent must
skip the critique LLM call entirely, and a turn whose whole-turn deadline
(`turn_deadline.py`) has already passed must not pay for a refinement
judgement it has no time left to act on. All three checks live in
`_skip_critique_llm_call`, a `before_agent_callback`, which is deterministic
Python, precisely because the whole point is that none of these three
decisions may be a model judgement - "the model chose to approve fast" is not
the same guarantee as "no LLM call happened here".

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
from src.research_agent import token_budget, tool_budget, turn_deadline
from src.services import genai_client

CRITIQUE_AGENT_NAME = "critique_agent"

# Session-state key holding the artefact create_canvas rendered this cycle, so
# the critique reviews the deliverable rather than the covering note that
# accompanies it. Written by agent.py's after_tool_callback, read by this
# module's INSTRUCTION via {last_artefact?}, cleared per turn by
# reset_turn_state below.
#
# It lives in state rather than being passed through draft_answer because
# draft_answer is research_agent's output_key - ADK owns what goes in it, and
# an artefact turn's draft is legitimately a short note. Overwriting it would
# also corrupt what the evaluation stores as the cycle's draft, which is the
# input to check_wasted_cycle's similarity comparison.
LAST_ARTEFACT_KEY = "last_artefact"


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
    # Cleared per turn like everything else here. Without this a session that
    # asked for a report and then asked an ordinary question would show the
    # previous turn's artefact to the critique, which would judge a plain
    # answer against a stale document - the same cross-turn leak this whole
    # callback exists to prevent.
    state[LAST_ARTEFACT_KEY] = ""
    # research_agent's per-turn tool-call budget rides on this same hook
    # rather than its own: this is the only callback in the system that fires
    # once per turn instead of once per refinement cycle, and a budget that
    # refilled each cycle would not bound a loop that re-runs the same
    # searches - see tool_budget.py's module docstring.
    state[tool_budget.STATE_KEY] = {}
    # Same turn-scoping requirement as STATE_KEY directly above, and the same
    # reason: report_gap's stop condition (tool_budget.py, "report_gap as a
    # second, independent stop condition") must not leak into the next turn -
    # left set from a prior turn's gap it would refuse that new turn's first
    # evidence call for a question it never reported anything against.
    state[tool_budget.STATE_KEY_GAP_REPORTED] = False
    # Same turn-scoping requirement again, for declare_plan's gate
    # (tool_budget.py, "declare_plan as a third, independent gate"). This is
    # exactly the class of bug ADR-0021's fourth supporting decision names: a
    # key that must be turn-scoped and is put on the wrong hook fails
    # silently, with no error and no warning. Left set from a prior turn's
    # plan, this would refuse a brand-new turn's very first evidence call
    # against a plan declared for a different question entirely - the same
    # failure shape STATE_KEY_GAP_REPORTED's comment above describes, and the
    # reason this line lives on the same hook as that one rather than on
    # something that only fires once per session.
    state[tool_budget.STATE_KEY_DECLARED_SOURCES] = []
    # The fact-level half of the same plan, and the record of which fact each
    # evidence call served (2026-08-07, audit findings 3 and 4). Both are
    # turn-scoped for exactly the reason the line above is: a fact plan
    # surviving into the next turn would gate that turn's calls against the
    # previous question's facts, and every one of them would be refused as
    # not-in-the-plan. The call record surviving would be worse still - it
    # would lock sources against facts the new question never declared.
    state[tool_budget.STATE_KEY_DECLARED_FACTS] = {}
    state[tool_budget.STATE_KEY_FACT_CALLS] = {}
    # Deliberately NOT reset here: token_budget.STATE_KEY. It is the one
    # counter in this system that is session-scoped rather than turn-scoped,
    # and clearing it on this hook - the hook that exists to make things
    # turn-scoped - would quietly turn a session ceiling into a per-turn one
    # that can never be reached. No error, no warning, just a bound that never
    # fires. See src/research_agent/token_budget.py.


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


# Non-evidence tools subtracted from a cycle's tool set before the
# financial-only comparison in _skip_critique_llm_call, below.
#
# Deliberately derived from tool_budget.OUTPUT_TOOLS rather than a second,
# hand-written list: this whole fix exists because declare_plan became a
# mandatory tool on every turn (2026-08-06, agent.py's INSTRUCTION step 1)
# and the exact-equality check below (`_tools_used_this_cycle(...) ==
# {"get_financial_data"}`) was never updated to know that - a financial-only
# cycle now always also calls declare_plan, the equality stopped holding, and
# the deterministic skip this module was specifically built to guarantee (see
# the module docstring, and ADR-0010) silently became unreachable. See
# agent_docs/audit.md, Finding 8.
#
# Deriving from OUTPUT_TOOLS rather than duplicating a third list means a
# FUTURE mandatory non-evidence tool is absorbed automatically the moment it
# is added there for the tool-call ceiling's sake (tool_budget.py) - which it
# must be, or the ceiling breaks the same way declare_plan broke this check -
# rather than needing a second, easily-forgotten edit here. It is not
# schema.py's own OUTPUT_TOOLS: that copy is owned by the evaluation code and
# deliberately kept independent so scoring a stored run never has to import
# the agent package (see schema.py's comment on OUTPUT_TOOLS) - a concern
# that does not apply here, since critique.py already imports tool_budget for
# its state keys.
#
# create_canvas is carved back OUT of that derived set, on purpose: it is
# exempt from the tool-call ceiling for a completely different reason (it
# retrieves nothing, see tool_budget.py) but it is not bookkeeping - it
# renders the actual deliverable, and ADR-0016 requires the critic to review
# that rendered artefact on a Canvas turn. Subtracting it here would let a
# turn that rendered a whole report from a single financial figure skip
# critique entirely, which is exactly the review ADR-0016 added. So the
# default assumption below is "nothing to critique", not "safe to skip" - any
# future non-evidence tool that, like create_canvas, produces content for the
# critique to review needs the same explicit carve-out, or it will silently
# defeat this check the same way declare_plan defeated the old one.
_TOOLS_WITH_NOTHING_TO_CRITIQUE: frozenset[str] = tool_budget.OUTPUT_TOOLS - {"create_canvas"}


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

    Three independent reasons to skip, checked every cycle, in this order:
    - The turn's whole-turn deadline (turn_deadline.stamp_turn_deadline, the
      same clock research_agent's own before_agent_callback enforces) has
      already passed. Checked first, and with its own message rather than
      turn_deadline.enforce_turn_deadline's Content reused verbatim, because
      that message says "no answer was produced this turn" - true there,
      because it fires before that cycle's LLM call ever ran, but false here:
      by the time critique_agent's callback runs, research_agent has already
      written a real draft_answer this cycle. What is being skipped is one
      refinement pass, not the turn's only answer - and letting critique run
      anyway would be actively worse than skipping it: a genuine follow-up
      would send the loop back to research_agent, whose OWN deadline gate
      would then fire on the very next cycle and overwrite this cycle's
      perfectly good draft_answer with ITS "no answer was produced" message,
      turning a real answer into a reported timeout for no benefit to the
      user. Skipping here is not a judgement that the draft is complete (the
      model never looked) - it is the same "termination is the default
      outcome" property the rest of this module is built around, applied to
      the one case where continuing is strictly worse than stopping. See
      agent_docs/audit.md, Finding 11b, and turn_deadline.py's module
      docstring for the full reasoning on the shared clock.
    - The per-request budget (session state "critique_budget", defaulting to
      config.DEFAULT_CRITIQUE_BUDGET when the request didn't set one) is 0
      or has already been spent by an earlier cycle's real critique call.
      This is what makes a budget of 0 reproduce the pre-phase-4 single-cycle
      behaviour exactly: the very first check on the very first cycle finds
      0 >= 0 and escalates before anything resembling a critique runs.
    - The just-completed research cycle's only EVIDENCE tool call was
      get_financial_data - a single live-price lookup has no citation to
      omit and no sub-question left to ask, so critiquing it is pure
      overhead on a path this project already measured close to its latency
      target. Compared after subtracting _TOOLS_WITH_NOTHING_TO_CRITIQUE (see
      its own comment, immediately above) so a mandatory bookkeeping call
      like declare_plan cannot defeat this the way it used to.

    Returning this event, authored by critique_agent (ADK sets `author=
    self.name` on a before_agent_callback's returned Content - see
    turn_deadline.py's module docstring for where that is verified), never
    touches research_agent's own final-response event for this cycle. Every
    consumer that follows CLAUDE.md's "Reading a turn's answer" rule matches
    on `event.author == research_agent.name`, so this event - like the
    budget-exhausted and financial-only skip messages that already existed
    before this deadline check - is simply invisible to that rule, not a
    second candidate answer it could be confused with.
    """
    # original_query/critique_iterations_used are already turn-fresh by the
    # time this runs - reset_turn_state (the loop's own before_agent_callback)
    # sets them before research_agent's first cycle of this turn.
    state = callback_context.state

    if turn_deadline.deadline_exceeded(callback_context):
        callback_context.actions.escalate = True
        return types.Content(
            role="model",
            parts=[
                types.Part(
                    text=(
                        "Skipping critique: this turn's time budget is already spent. "
                        "Ending the loop with the answer already produced this cycle "
                        "rather than spending more time on a refinement pass."
                    )
                )
            ],
        )

    budget = state.get("critique_budget", config.DEFAULT_CRITIQUE_BUDGET)
    spent = state.get("critique_iterations_used", 0)
    budget_exhausted = spent >= budget

    cycle_tools = _tools_used_this_cycle(callback_context) - _TOOLS_WITH_NOTHING_TO_CRITIQUE
    financial_only = cycle_tools == {"get_financial_data"}

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

Artefact this cycle produced, if any (empty for most turns):
{last_artefact?}

If the artefact above is non-empty, the turn was asked for a deliverable and
THAT is the answer under review - the draft above it is only a covering note,
so judge completeness and citation against the artefact and not against the
note. An artefact that answers the question is complete even if the note
mentions almost nothing. Never raise a follow-up asking for something the
artefact already contains, and never ask for the artefact to be reformatted,
restructured, or restyled: form is not a gap.

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
- A part of the question the draft explicitly declines - it states plainly
  that the source checked does not cover it, and names that source. That
  part has been answered by being correctly declined, not left unaddressed:
  the source's not having it IS the answer. A follow-up chasing the exact
  figure just declined sends research back to a fact it has already
  confirmed is missing, wasting a cycle on a result that cannot change.

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
    # See genai_client.MODEL_CALL_TIMEOUT_MS: ADK's own client sets no
    # timeout, so an unbounded critique call would hang a turn indefinitely.
    http_options=types.HttpOptions(timeout=genai_client.MODEL_CALL_TIMEOUT_MS),
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
    # The session token ceiling, same pair as research_agent's - this agent
    # calls the model too, and a bound with a hole in it is not a bound.
    # See src/research_agent/token_budget.py.
    before_model_callback=token_budget.enforce_session_token_budget,
    after_model_callback=token_budget.accumulate_token_usage,
    generate_content_config=_GENERATE_CONTENT_CONFIG,
)
