"""Regression test for audit.md finding 9: `acceptable_routes` disagreeing
between the three places that read it.

`metrics.check_routing` is the authoritative scorer: it treats
`EvalQuestion.acceptable_routes` as whole alternative route sets, each a full
substitute for `expected_routes`, and passes a run whose `routes_used`
matches EITHER `expected_routes` OR exactly one of those alternatives. Before
this change, `preflight.needed_services` and `langfuse_sync._scores_for` both
still keyed off `expected_routes` alone, so a question with a legitimate
second route (`multi-web-and-financial` - `expected_routes: [financial,
web]`, `acceptable_routes: [[financial, news_agent]]`, real shape copied from
data/eval/questions.yaml) produced two independent failures documented in
agent_docs/audit.md:

1. A sweep filtered to just this question, with mcp-fetch up and news-agent
   down, reported "reachable" - `needed_services` only ever looked at
   `expected_routes` ([financial, web]), which needs mcp-fetch and (for the
   `web` route) nothing from compose, so a genuinely down news-agent was
   invisible to preflight even though the live planner reaches it on this
   question 8 of 8 runs (EVALUATION.md).
2. `--sync-langfuse` pushed `routing_correct=0.0` for a run that took the
   [financial, news_agent] alternative, because `_scores_for` compared
   `set(routes_used) == set(expected_routes)` directly instead of asking
   `check_routing`, disagreeing with the authoritative JSON (which
   `check_routing` correctly scores as a pass) on the very same run.

Both tests below are offline: no Vertex call, no live compose service, no
Langfuse client. They assert an INVARIANT (preflight's services are a
superset of what any candidate route needs; langfuse_sync's score agrees with
check_routing) rather than a hardcoded expected value, so they keep catching
drift between the three sites even if the question set or check_routing's own
logic changes later.
"""

from __future__ import annotations

from src.evaluation import preflight
from src.evaluation.langfuse_sync import _scores_for
from src.evaluation.metrics import check_routing
from src.evaluation.schema import CycleRecord, EvalQuestion, RunRecord

# Real shape of data/eval/questions.yaml's multi-web-and-financial, the only
# question in the set that carries acceptable_routes today (confirmed by
# grepping the file: one match). Built via from_dict, the same path
# run_eval.load_questions uses, rather than the dataclass constructor
# directly, so this test exercises the same RouteTarget coercion a real
# questions.yaml load does.
_QUESTION = EvalQuestion.from_dict(
    {
        "id": "multi-web-and-financial",
        "question": (
            "What is the current price of Bitcoin, and what has the European "
            "Central Bank most recently said about regulating crypto assets?"
        ),
        "expected_routes": ["financial", "web"],
        "acceptable_routes": [["financial", "news_agent"]],
        "max_tool_calls": 3,
    }
)


def test_needed_services_includes_acceptable_route_alternatives() -> None:
    """preflight must see news-agent as needed, not just mcp-fetch.

    `expected_routes` alone ([financial, web]) only ever implies mcp-fetch -
    `web` has no compose service (google_search needs nothing this preflight
    tracks). Before the fix, a sweep scoped to just this question would
    therefore never check news-agent at all, even though `acceptable_routes`
    names [financial, news_agent] as an equally correct route the live
    planner actually takes (EVALUATION.md: 8 of 8 runs). That is finding 9's
    first failure scenario, reproduced offline: no live service is contacted
    here, `needed_services` only inspects already-loaded EvalQuestion data.
    """
    services = preflight.needed_services([_QUESTION])

    assert "mcp-fetch" in services, (
        f"expected_routes' financial route needs mcp-fetch, got {services!r}"
    )
    assert "news-agent" in services, (
        "acceptable_routes' [financial, news_agent] alternative needs news-agent too - "
        f"a sweep on this question must not report 'reachable' with it down, got {services!r}"
    )


def test_needed_services_empty_for_a_question_with_no_alternatives() -> None:
    """The union must not over-trigger on ordinary questions.

    A question with no `acceptable_routes` (the default, empty list) should
    behave exactly as before: only what `expected_routes` names. Guards
    against a fix that widens the union so much it drags every service into
    every sweep, which would defeat the whole point of scoping preflight to
    the SELECTED questions (preflight.py's own module docstring).
    """
    kb_only = EvalQuestion.from_dict(
        {
            "id": "kb-only-smoke-question",
            "question": "What is IFC's stated mission?",
            "expected_routes": ["knowledge_base"],
        }
    )
    assert preflight.needed_services([kb_only]) == []


def _run_via(*tool_names: str) -> RunRecord:
    """A minimal single-cycle RunRecord that called exactly `tool_names`."""
    return RunRecord(
        question_id=_QUESTION.id,
        question=_QUESTION.question,
        arm="test-arm",
        rep=0,
        mode="live",
        trace_id="trace-acceptable-routes-test",
        cycles=[CycleRecord(index=0, tools_called=list(tool_names))],
        answer="Bitcoin is trading at $X; the ECB most recently said Y.",
    )


def test_langfuse_routing_score_agrees_with_check_routing_on_an_acceptable_alternative() -> None:
    """The score pushed to Langfuse must match the authoritative JSON's verdict.

    This is finding 9's second failure scenario: a run that took the
    [financial, news_agent] alternative is a PASS under `check_routing` (the
    scorer `run_eval.py`'s stored JSON is built from), so `--sync-langfuse`
    pushing routing_correct=0.0 for the same run would show a routing
    regression on the dashboard that the authoritative record does not agree
    with. Asserted as an equality between the two computations rather than a
    hardcoded 1.0/True, so this test keeps catching the two sites drifting
    apart even if check_routing's own semantics change later - which is
    exactly the property the audit asked for ("so the two never disagree").
    """
    record = _run_via("get_financial_data", "news_agent")

    authoritative = check_routing(_QUESTION, record).passed
    scores = dict((name, value) for name, value, _data_type in _scores_for(record, _QUESTION))
    pushed_to_langfuse = scores["routing_correct"] == 1.0

    assert authoritative is True, (
        "test setup assumption failed: [financial, news_agent] is exactly "
        "multi-web-and-financial's acceptable_routes entry and should pass check_routing"
    )
    assert pushed_to_langfuse == authoritative, (
        f"langfuse_sync scored routing_correct={scores['routing_correct']} "
        f"but metrics.check_routing says passed={authoritative} for the same run - "
        "the dashboard and the authoritative JSON disagree on identical evidence"
    )


def test_langfuse_routing_score_agrees_with_check_routing_on_a_genuine_miss() -> None:
    """The other direction: a real routing failure must still score as one.

    Guards against a fix that over-corrects by treating ANY route set as
    acceptable - a run that consulted only the knowledge base (no route this
    question names anywhere) must still fail both the authoritative check and
    the Langfuse score, and the two must still agree with each other.
    """
    record = _run_via("search_documents")

    authoritative = check_routing(_QUESTION, record).passed
    scores = dict((name, value) for name, value, _data_type in _scores_for(record, _QUESTION))
    pushed_to_langfuse = scores["routing_correct"] == 1.0

    assert authoritative is False, (
        "test setup assumption failed: search_documents alone matches neither "
        "expected_routes nor the acceptable_routes alternative"
    )
    assert pushed_to_langfuse == authoritative, (
        f"langfuse_sync scored routing_correct={scores['routing_correct']} "
        f"but metrics.check_routing says passed={authoritative} for the same run"
    )
