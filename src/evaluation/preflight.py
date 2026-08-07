"""Fail-fast checks for run_eval.py's live/record-mode sweeps.

Why this exists: `data/eval/fixtures` aside, `--mode live`/`record` calls the
two external compose services (`mcp-fetch`, `news-agent`) for real. When one
is not running, the failure does not surface here - it surfaces deep inside
the agent turn as a routing or content failure (financial_data.py's own
`_unreachable`/news_agent.py's `_unreachable` degrade gracefully rather than
raising), which reads as an agent defect across every affected run in a
280-run sweep rather than as "you forgot to start a container". Checking once,
before any run executes, turns that into one readable failure instead of many
confusing ones.

Deliberately scoped to what the SELECTED questions need, not "both services,
always": `EvalQuestion.expected_routes` (and `acceptable_routes`, its
alternative sets - see `needed_services`) is already-loaded data by the time a
sweep starts (see run_eval.load_questions), so mapping FINANCIAL ->
`mcp-fetch` and NEWS_AGENT -> `news-agent` costs nothing extra to compute and
means a `--tags kb` smoke run is never blocked by a `news-agent` it will never
call.

Not run in replay mode at all (see run_eval.main_async): replay's whole point
is that `replay.ToolFixtureSweep` patches every evidence-gathering tool so
`real_call` is never invoked (replay.py's own module docstring, "Replay: real_call
is never invoked, by construction") - no live service call happens regardless
of what is or is not running, so a preflight check there would be checking
something the run never depends on.
"""

from __future__ import annotations

import asyncio

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src import config
from src.evaluation.schema import EvalQuestion, RouteTarget

# Short and fixed, deliberately NOT config.MCP_FETCH_TIMEOUT_S (30s) or
# config.NEWS_AGENT_TIMEOUT_S (20s) - those bound a real fetch/RPC inside a
# turn; this only needs to prove a service is listening at all, before any
# run starts, so a couple of seconds is enough even against a hung process.
PREFLIGHT_TIMEOUT_S = 3.0

# Which RouteTarget needs which compose service, and therefore which service
# name belongs in the fix-it command. Named once here so `needed_services`
# and the failure message can't drift apart from each other.
_ROUTE_TO_SERVICE: dict[RouteTarget, str] = {
    RouteTarget.FINANCIAL: "mcp-fetch",
    RouteTarget.NEWS_AGENT: "news-agent",
}

# Fixed order (matches docker-compose.yml's service order) so the printed
# fix-it command reads the same way on every failing sweep instead of
# depending on dict/set iteration order.
_SERVICE_ORDER = ("mcp-fetch", "news-agent")


def _first_leaf(exc: BaseException) -> BaseException:
    """The innermost exception, unwrapping nested ExceptionGroups.

    A timeout against an address that never responds (as opposed to a
    refused connection) wraps its `TimeoutError` inside more than one nested
    `TaskGroup`/`ExceptionGroup` layer here - one unwrap left a useless
    "unhandled errors in a TaskGroup" message with no indication of what
    actually failed, checked against a real hung-address probe rather than
    assumed. Same helper as financial_data.py's `_first_leaf`, duplicated
    rather than imported - this module owns no dependency on src/tools/.
    """
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


def needed_services(questions: list[EvalQuestion]) -> list[str]:
    """Which compose services the selected questions actually exercise.

    Cheap: EvalQuestion.expected_routes is already-loaded data, not something
    that needs a live call to determine.

    Unions `expected_routes` with EVERY alternative set in `acceptable_routes`
    (audit.md finding 9), not `expected_routes` alone. `acceptable_routes`
    holds whole alternative route sets that `metrics.check_routing` accepts as
    equally correct substitutes (schema.py's own framing), and this function
    runs before a single question has actually been asked - there is no way
    to know in advance which of several equally-correct routes a live planner
    will pick for a given run. "Might reach" is therefore the right union, not
    an intersection or `expected_routes` alone: checking only the declared
    routes leaves exactly the gap the audit measured on
    multi-web-and-financial (`expected_routes: [financial, web]`,
    `acceptable_routes: [[financial, news_agent]]`) - a sweep with mcp-fetch
    up and news-agent down would report "reachable" from `expected_routes`
    alone, then silently degrade on every run where the live planner takes the
    news_agent alternative, which is precisely the failure this preflight
    exists to catch before any run starts. The cost of the wider union is
    just checking one extra service on the rare question that declares
    alternatives at all (one question in the set today) - cheap next to a
    280-run sweep degrading silently.
    """
    services: set[str] = set()
    for question in questions:
        for route_set in (question.expected_routes, *question.acceptable_routes):
            for route in route_set:
                service = _ROUTE_TO_SERVICE.get(route)
                if service is not None:
                    services.add(service)
    return [s for s in _SERVICE_ORDER if s in services]


async def _check_mcp_fetch() -> str | None:
    """None if the MCP fetch server completes an MCP handshake; else why not.

    A real `initialize()` against config.MCP_FETCH_URL, not a bare TCP probe:
    the compose service is a stdio-to-HTTP proxy in front of the reference
    `fetch` server (ADR-0020), so a socket that merely accepts a connection
    does not prove the proxy is forwarding to a working MCP session behind
    it. No `fetch` tool call is made - only the same handshake
    financial_data.get_financial_data itself opens before every real call -
    so this never touches a live Yahoo Finance page.
    """
    try:
        async with streamablehttp_client(config.MCP_FETCH_URL, timeout=PREFLIGHT_TIMEOUT_S) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=PREFLIGHT_TIMEOUT_S)
    except BaseExceptionGroup as group:
        # streamablehttp_client runs its reader/writer in an anyio task group,
        # so a refused connection (or a timeout against an address that never
        # responds at all) arrives wrapped rather than raised directly - the
        # same shape financial_data.py already unwraps, for the same reason
        # (see that module's _unreachable).
        leaf = _first_leaf(group)
        return f"{type(leaf).__name__}: {leaf}"
    except Exception as exc:  # noqa: BLE001 - any failure here means "not ready"
        return f"{type(exc).__name__}: {exc}"
    return None


async def _check_news_agent() -> str | None:
    """None if the News Agent serves its agent card; else why not.

    Hits the card, not an RPC: `src/tools/news_agent.py` resolves this same
    URL lazily on first real use, and a servable card is what "the service is
    up" means for an A2A agent (see that module's docstring on why the card
    is fetched lazily rather than at import).
    """
    url = f"{config.NEWS_AGENT_URL}/.well-known/agent-card.json"
    try:
        async with httpx.AsyncClient(timeout=PREFLIGHT_TIMEOUT_S) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any failure here means "not ready"
        return f"{type(exc).__name__}: {exc}"
    return None


_CHECKS = {
    "mcp-fetch": _check_mcp_fetch,
    "news-agent": _check_news_agent,
}


async def run_preflight(questions: list[EvalQuestion]) -> None:
    """Raise SystemExit before any run executes if a needed service is down.

    Only checks what `questions` (the already-filtered selection for this
    sweep) actually needs - see `needed_services`. Called from
    run_eval.main_async after loading/filtering questions and before
    run_sweep starts, and skipped entirely for --mode replay or
    --skip-preflight (see main_async).
    """
    services = needed_services(questions)
    if not services:
        # Visible on purpose, not a silent early return: if a future change
        # to needed_services or TOOL_TO_ROUTE ever made this branch fire
        # unconditionally, a quiet no-op here would mean the check stopped
        # doing anything and nobody would notice until the next 280-run
        # sweep failed deep inside a turn - the exact failure mode this
        # module exists to prevent.
        print("Preflight: no external service needed for this question selection.", flush=True)
        return

    reasons = await asyncio.gather(*(_CHECKS[service]() for service in services))
    down = [(service, reason) for service, reason in zip(services, reasons) if reason is not None]
    if not down:
        print(f"Preflight: {', '.join(services)} reachable.", flush=True)
        return

    detail = "; ".join(f"{service} ({reason})" for service, reason in down)
    fix_services = " ".join(service for service, _ in down)
    raise SystemExit(
        f"Preflight failed - required service(s) not reachable: {detail}.\n"
        f"Fix with: docker compose up -d {fix_services}\n"
        "If you just started it, mcp-fetch's stdio-to-HTTP proxy can take a "
        f"few seconds to come up (untested here - the {PREFLIGHT_TIMEOUT_S:.0f}s "
        "check timeout was not verified against a cold start, only a warm "
        "one) - wait a moment and retry before assuming this is a real outage.\n"
        "(or pass --skip-preflight to bypass this check, e.g. for a deliberate "
        "negative test against a live service)."
    )
