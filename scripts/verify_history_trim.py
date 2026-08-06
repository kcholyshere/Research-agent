"""Manual verification for src/research_agent/history_trim.py (the ADR-0021
follow-up: trim research_agent's resent history to the last N turns).

Not a pytest suite, same reason scripts/verify_agent.py and
scripts/verify_turn_timeout.py aren't (no tests/ directory - see
verify_agent.py's docstring). Five things this script has to actually show,
not just claim:

1. The real llm_request.contents structure across a multi-turn session, and
   that the trim keeps whole turns (question, every tool call/response,
   final answer, critique remark) rather than cutting one apart - a probe
   installed on research_agent.before_model_callback prints the contents it
   is about to send, after the real trim callback has already run.
2. What the trim actually removes, measured deterministically. An earlier
   version of this script compared prompt_token_count across TWO separate
   live sessions (trim on vs off) - rejected on review: turns inside the
   retention window came back with DIFFERENT token counts between the two
   runs despite identical inputs, because each session makes its own live
   web_search_agent calls and gets its own live grounding results back. That
   cross-session noise was large enough to swallow the very effect being
   measured. This version instead counts llm_request.contents - via the
   installed google-genai client's count_tokens, verified against the
   installed package below - immediately before and immediately after the
   real trim callback runs, ON THE SAME REQUEST, inside ONE session. No
   second session, no live-content variance: the only thing that can differ
   between "before" and "after" is the trim itself.
3. Whether the trim fires on every research_agent model call within a turn
   or only the first. A turn makes several model calls (declare_plan, one
   per evidence tool, the final synthesis, sometimes a refinement cycle) -
   a dedicated run with an artificially small window (MAX_HISTORY_TURNS
   patched to 1, so firing starts within a few cheap turns instead of
   needing the real window's five) shows the trim recomputing on every one
   of a turn's calls, and - the part worth checking rather than assuming -
   that its effect (how many contents/tokens it drops) is IDENTICAL across
   every call within one turn, not progressively deeper call over call.
4. token_budget.STATE_KEY (the session-scoped cumulative counter ADR-0021
   added) still only ever climbs, turn over turn - the hard constraint this
   whole change must not disturb.
5. A genuine cross-turn follow-up ("what about the year before?") still
   answered correctly with trimming on - ADR-0021's own stated objection to
   this option ("silently drops context the agent may need mid-conversation")
   demonstrated as NOT happening for a question inside the retained window.

Run with: uv run python -m scripts.verify_history_trim
"""

import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from src import config
from src.research_agent import token_budget
from src.research_agent.agent import research_agent, root_agent  # noqa: F401 - import order = instrumentation
from src.research_agent.history_trim import _opens_a_turn, trim_history
from src.services import genai_client

# The installed google-genai client, the same one src/services/genai_client.py
# gives every other module in this project - checked directly (not assumed
# from docs) that Client.models.count_tokens(model=, contents=) works against
# Vertex for gemini-3.5-flash and accepts Content made of function_call/
# function_response parts (most of what's actually in these contents lists),
# returning a real CountTokensResponse.total_tokens.
_count_tokens_client = genai_client.get_client()


def _count_tokens(contents: list[genai_types.Content]) -> int:
    """Real token count of `contents` alone, via the model's own tokenizer -
    not a proxy. Deliberately contents-only rather than the whole LlmRequest
    (system instruction, tool declarations): those never change between the
    "before" and "after" measurement below, so isolating contents isolates
    exactly what the trim changed.
    """
    if not contents:
        return 0
    return _count_tokens_client.models.count_tokens(
        model=config.GEMINI_MODEL, contents=contents
    ).total_tokens


def _turn_number(contents: list[genai_types.Content]) -> int:
    """Which user turn this request is mid-way through, by counting turn-
    opening boundaries - the same structural test history_trim.py's own
    `_opens_a_turn` uses to decide what to keep."""
    return sum(1 for c in contents if _opens_a_turn(c))

APP_NAME = "verify-history-trim"
USER_ID = "verify-history-trim"

# Five plain KB questions - one more turn than config.MAX_HISTORY_TURNS (3)
# plus the in-progress one, so turn 5 is the first point a trim can fire.
# Reused from the same curated set verify_agent.py draws on.
#
# Deliberately not six: an earlier draft added a sixth question here and hit
# config.MAX_SESSION_TOKENS's refusal wall mid-run - turn 4 alone made six
# real model calls (an unusually search-heavy live run) and the session's
# real cumulative cost was already past 200,000 by the end of turn 5, so a
# turn 6 got refused before a single model call, and refused calls never
# reach this script's probes (token_budget.enforce_session_token_budget
# short-circuits ahead of them in the callback list - see below). Five turns
# is what point 2 needs (one clean trim event); the "fires on every call,
# stable within a turn" claim (point 3) is checked separately below with an
# artificially small window instead, precisely to not depend on a long
# multi-call turn landing this close to the session ceiling.
QUESTIONS = [
    "What was IFC's Net Income for the fiscal year ending June 30, 2024?",
    "What was the total value of IFC's assets as of June 30, 2023?",
    "What is the stated mission of the International Finance Corporation?",
    "What was IFC's total value of Loan Investments as of June 30, 2024?",
    "What is IFC's official headquarters location?",
]


def _content_summary(c: genai_types.Content) -> str:
    kinds = []
    for part in c.parts or []:
        if part.function_call:
            kinds.append(f"function_call:{part.function_call.name}")
        elif part.function_response:
            kinds.append(f"function_response:{part.function_response.name}")
        elif part.text:
            preview = part.text.strip().replace("\n", " ")[:50]
            kinds.append(f"text:{preview!r}")
        else:
            kinds.append("other")
    return f"role={c.role:6s} parts=[{', '.join(kinds)}]"


async def _run_turn(runner: InMemoryRunner, session_id: str, question: str) -> str:
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
    final_text = "(no response)"
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
        state_delta={"critique_budget": 0},
    ):
        if (
            event.author == research_agent.name
            and event.is_final_response()
            and event.content
            and event.content.parts
        ):
            final_text = "".join(p.text or "" for p in event.content.parts)
    return final_text


# ---------------------------------------------------------------------------
# 1. Contents structure, before/after the trim, on a real multi-turn session.
# ---------------------------------------------------------------------------


async def verify_contents_structure() -> None:
    print(f"\n{'=' * 80}\n[1] llm_request.contents structure across a multi-turn session\n{'-' * 80}")

    seen: list[tuple[int, list[genai_types.Content]]] = []

    def _probe(callback_context, llm_request):
        seen.append((len(llm_request.contents), list(llm_request.contents)))
        return None

    original = research_agent.before_model_callback
    assert isinstance(original, list)
    research_agent.before_model_callback = [*original, _probe]  # trim already ran by here
    try:
        runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        # 5, not 4: config.MAX_HISTORY_TURNS=3 keeps 3 past turns + the current
        # one = 4 turns' worth untouched, so a 5th turn is the first point a
        # shrink can actually fire - see the "<-- trim fired" marker below.
        for q in QUESTIONS[:5]:
            await _run_turn(runner, session.id, q)
    finally:
        research_agent.before_model_callback = original

    # Print the FIRST call of each turn (post-trim) plus the very last call
    # overall, so the boundary behaviour is visible without a huge dump: each
    # first-of-turn call is where a trim (if any) has just taken effect.
    print(f"config.MAX_HISTORY_TURNS = {config.MAX_HISTORY_TURNS}")
    prev_len = None
    for i, (n, contents) in enumerate(seen, start=1):
        shrank = prev_len is not None and n < prev_len
        marker = "  <-- shorter than the previous call (trim fired)" if shrank else ""
        print(f"call #{i}: {n} contents{marker}")
        prev_len = n
    print("\nFull dump of the LAST call's contents (post-trim):")
    for i, c in enumerate(seen[-1][1]):
        print(f"  [{i}] {_content_summary(c)}")
    # A trimmed request must never start mid-turn: its first entry has to be
    # a genuine turn boundary (a real question, not a tool response or the
    # "For context:" wrapper) - assert this structurally rather than eyeball it.
    from src.research_agent.history_trim import _opens_a_turn

    assert _opens_a_turn(seen[-1][1][0]), (
        "trimmed contents did not start on a turn boundary - a tool call/response "
        "pair or a critique remark was cut apart"
    )
    print("PASS: trimmed request's first content is a genuine turn boundary.")


# ---------------------------------------------------------------------------
# 2, 3 & 4. What the trim removes (before/after, same request, one session),
# whether it fires every call, and the session token counter.
# ---------------------------------------------------------------------------


async def _run_instrumented(questions: list[str]) -> tuple[list[dict], list[int]]:
    """Runs `questions` in one fresh session with the trim callback bracketed
    by two probes in the SAME before_model_callback list, so "before" and
    "after" are the same request, not two different live calls. Returns the
    per-call rows and, for each turn, token_budget.STATE_KEY right after it.
    """
    rows: list[dict] = []
    pending: dict = {}

    def _before_probe(callback_context, llm_request):
        # Runs after token_budget.enforce_session_token_budget (which never
        # touches contents) and before history_trim.trim_history - so this
        # is contents exactly as trim_history is about to see them. A call
        # that enforce_session_token_budget refuses never reaches here (it
        # short-circuits the whole before_model_callback chain), which is
        # why QUESTIONS is sized to stay well clear of MAX_SESSION_TOKENS -
        # see that constant's comment above.
        pending["turn"] = _turn_number(llm_request.contents)
        pending["contents_before"] = len(llm_request.contents)
        pending["tokens_before"] = _count_tokens(llm_request.contents)
        return None

    def _after_probe(callback_context, llm_request):
        # Runs immediately after trim_history, same before-call chain, same
        # llm_request object, same request - only trim_history could have
        # changed anything between this and the probe above.
        contents_after = len(llm_request.contents)
        tokens_after = _count_tokens(llm_request.contents)
        rows.append({
            "call": len(rows) + 1,
            "turn": pending["turn"],
            "contents_before": pending["contents_before"],
            "contents_after": contents_after,
            "tokens_before": pending["tokens_before"],
            "tokens_after": tokens_after,
            "fired": contents_after < pending["contents_before"],
            "dropped_contents": pending["contents_before"] - contents_after,
            "dropped_tokens": pending["tokens_before"] - tokens_after,
        })
        return None

    original_before = research_agent.before_model_callback
    assert isinstance(original_before, list) and original_before[1] is trim_history
    research_agent.before_model_callback = [
        original_before[0],  # token_budget.enforce_session_token_budget
        _before_probe,
        trim_history,
        _after_probe,
    ]
    counter_after_turn: list[int] = []
    try:
        runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        for q in questions:
            await _run_turn(runner, session.id, q)
            state = await runner.session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session.id
            )
            counter_after_turn.append(state.state.get(token_budget.STATE_KEY, 0))
    finally:
        research_agent.before_model_callback = original_before

    return rows, counter_after_turn


def _print_rows(rows: list[dict]) -> None:
    header = f"{'call':>4} {'turn':>4} {'contents before->after':>23} {'tokens before->after':>21}  fired"
    print(header)
    for r in rows:
        contents_str = f"{r['contents_before']:>3} -> {r['contents_after']:<3}"
        tokens_str = f"{r['tokens_before']:>5} -> {r['tokens_after']:<5}"
        print(f"{r['call']:>4} {r['turn']:>4} {contents_str:>23} {tokens_str:>21}  {r['fired']}")


async def verify_trim_savings_and_counter() -> None:
    print(
        f"\n{'=' * 80}\n[2+4] Trim savings (before/after, same request, one "
        f"session) and the session counter\n{'-' * 80}"
    )
    rows, counter_after_turn = await _run_instrumented(QUESTIONS)

    print(f"config.MAX_HISTORY_TURNS = {config.MAX_HISTORY_TURNS}\n")
    _print_rows(rows)

    # The trim must never make a request LONGER, and once it starts firing
    # (the session has grown past the retained window) it must never go back
    # to not firing - turn count only grows across a session, so the
    # "len(turn_starts) > keep_segments" condition, once true, stays true.
    for r in rows:
        assert r["contents_after"] <= r["contents_before"], (
            f"call {r['call']}: trim GREW the request ({r['contents_before']} -> "
            f"{r['contents_after']} contents) - it must only ever shorten or leave alone"
        )
    first_fired = next((r["call"] for r in rows if r["fired"]), None)
    assert first_fired is not None, (
        "trim never fired in this run - increase the number of QUESTIONS "
        "past config.MAX_HISTORY_TURNS so there is something to trim"
    )
    for r in rows:
        if r["call"] > first_fired:
            assert r["fired"], (
                f"call {r['call']} did not fire even though call {first_fired} did - "
                "the trim should stay active for the rest of the session, not lapse"
            )
    dropped = next(r for r in rows if r["call"] == first_fired)
    print(
        f"\nPASS: trim never lengthens a request; first fired at call {first_fired} "
        f"(dropped {dropped['dropped_contents']} contents / "
        f"{dropped['dropped_tokens']} tokens from that one request) and stayed "
        "active every call afterwards."
    )

    # The hard constraint: the cumulative session token counter must only
    # ever climb - trimming the REQUEST must never disturb token_budget's
    # accumulated total, which is charged from the response actually
    # received, independently of anything this module does.
    for i in range(1, len(counter_after_turn)):
        assert counter_after_turn[i] >= counter_after_turn[i - 1], (
            f"session token counter went DOWN between turn {i} and {i + 1} "
            f"({counter_after_turn[i - 1]:,} -> {counter_after_turn[i]:,}) - trimming "
            "the request must never disturb token_budget's accumulated total"
        )
    print(
        f"PASS: cumulative session token counter climbed monotonically across "
        f"all {len(counter_after_turn)} turns: "
        + " -> ".join(f"{c:,}" for c in counter_after_turn)
    )


async def verify_fires_every_call_and_stable() -> None:
    """Whether the trim recomputes on every research_agent model call within
    a turn, or only the first - and, if every call, whether it keeps cutting
    deeper each time or holds a stable boundary.

    Uses an artificially small window (MAX_HISTORY_TURNS=1, patched for this
    run only) rather than the real one: the real window needs 5 turns before
    it fires even once (see QUESTIONS), and getting a SECOND fired call
    inside one turn needs a turn with multiple model calls after that point -
    not guaranteed on any particular live run, and one attempt at forcing it
    (a sixth question) is what hit the session token ceiling documented
    above. A window of 1 fires by turn 3, in a short, cheap, dedicated
    session, and every one of a turn's several calls (declare_plan, an
    evidence tool, synthesis) lands after that point - guaranteeing the
    multi-call-after-firing case this check needs, deterministically.
    """
    print(f"\n{'=' * 80}\n[3] Trim firing on every call within a turn, not just the first\n{'-' * 80}")

    from unittest import mock

    with mock.patch.object(config, "MAX_HISTORY_TURNS", 1):
        print(f"config.MAX_HISTORY_TURNS patched to {config.MAX_HISTORY_TURNS} for this run only")
        rows, _ = await _run_instrumented(QUESTIONS[:4])

    _print_rows(rows)

    # The probe runs on EVERY before_model_callback invocation - so "fires"
    # here means "recomputed the boundary and found something to cut", not
    # "ran at all" (it always runs; see history_trim.trim_history's own
    # short-circuit for when there is nothing to cut).
    fired_calls = [r["call"] for r in rows if r["fired"]]
    assert len(fired_calls) >= 2, (
        f"expected at least two fired calls to compare (got {len(fired_calls)}) - "
        "increase QUESTIONS[:4] or shrink MAX_HISTORY_TURNS further so a later "
        "turn's multiple calls all land after the trim starts firing"
    )
    print(f"\nfired on calls {fired_calls} - recomputed and cut something on more than one call, not just the first.")

    by_turn: dict[int, list[dict]] = {}
    for r in rows:
        by_turn.setdefault(r["turn"], []).append(r)
    checked_a_multi_call_turn = False
    for turn, turn_rows in by_turn.items():
        fired_rows = [r for r in turn_rows if r["fired"]]
        if len(fired_rows) < 2:
            continue
        checked_a_multi_call_turn = True
        dropped_contents = {r["dropped_contents"] for r in fired_rows}
        dropped_tokens = {r["dropped_tokens"] for r in fired_rows}
        assert len(dropped_contents) == 1, (
            f"turn {turn}: dropped content count varied across its own calls "
            f"({sorted(dropped_contents)}) - the trim is cutting progressively "
            "deeper within a single turn instead of holding a stable boundary"
        )
        assert len(dropped_tokens) == 1, (
            f"turn {turn}: dropped token count varied across its own calls "
            f"({sorted(dropped_tokens)}) - same problem, measured in tokens"
        )
        print(
            f"turn {turn}: {len(fired_rows)} calls after the trim was active, all "
            f"dropping exactly {dropped_contents.pop()} contents / "
            f"{dropped_tokens.pop()} tokens - stable, not progressive."
        )
    assert checked_a_multi_call_turn, (
        "no turn had 2+ fired calls to compare - the stability claim was not "
        "actually exercised this run"
    )
    print(
        "\nPASS: the trim recomputes on every research_agent model call, not just "
        "the first of a turn, and its effect is stable across a turn's calls "
        "rather than cutting progressively deeper."
    )


# ---------------------------------------------------------------------------
# 5. A genuine cross-turn follow-up, still answered correctly.
# ---------------------------------------------------------------------------


async def verify_followup_question() -> None:
    print(f"\n{'=' * 80}\n[5] Follow-up question depending on the previous turn's answer\n{'-' * 80}")
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    q1 = "What was IFC's Net Income for the fiscal year ending June 30, 2024?"
    a1 = await _run_turn(runner, session.id, q1)
    print(f"Q1: {q1}\nA1: {a1}")

    q2 = "What about the year before?"
    a2 = await _run_turn(runner, session.id, q2)
    print(f"\nQ2: {q2}\nA2: {a2}")
    print(
        "\n(eyeball: A2 should answer FY2023's net income, understood from A1's "
        "context, not ask 'the year before what?' - this turn is well inside "
        f"the retained MAX_HISTORY_TURNS={config.MAX_HISTORY_TURNS} window.)"
    )


async def main() -> None:
    await verify_contents_structure()
    await verify_trim_savings_and_counter()
    await verify_fires_every_call_and_stable()
    await verify_followup_question()
    print(f"\n{'=' * 80}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
