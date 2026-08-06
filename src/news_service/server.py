"""The News Agent as an independent A2A server (phase 5).

Transport: the A2A protocol, via ADK's own `to_a2a`. This replaces an earlier
hand-rolled FastAPI `POST /news` endpoint - see ADR-0017, superseded. That
version was written on the finding that `a2a` was not installed and concluded
HTTP was an acceptable substitute because the process boundary is real either
way. The conclusion was wrong: phase 5 names A2A, so the protocol IS the
deliverable, and "not installed" was the task rather than a reason to
substitute. `google.adk.a2a` ships inside google-adk 2.5.0 already; only the
`a2a-sdk` dependency behind it was missing, and it is one extra
(`google-adk[a2a]`, now declared in pyproject.toml).

What `to_a2a` gives us that the FastAPI version had to invent:

- An **agent card** at `/.well-known/agent-card.json`, built automatically from
  the agent's own name, description and tools. That is what makes this
  discoverable rather than merely reachable - a client points at the card and
  learns what the agent can do, instead of a human reading our source to find
  out the path is `/news` and the body key is `topic`.
- A **task lifecycle**, so a long call is a task with state rather than one
  blocking request that either returns or times out.
- **Typed message conversion** both ways, so the client receives ADK events
  rather than a JSON blob it has to reshape.

The old endpoint's whole body - a Pydantic request model, an InMemoryRunner, and
a hand-written event scan to find the agent's final response - is gone, because
`to_a2a` builds the runner and does the conversion. Note what that removes:
picking the right final response out of a multi-agent event stream is the one
thing CLAUDE.md flags as having been got silently wrong three times on this
project, and this file no longer does it by hand.

`to_a2a` returns a Starlette app, so uvicorn serves it exactly as before.

Run with:

    uv run python -m src.news_service.server

Binds 0.0.0.0:8001 by default; override with the NEWS_AGENT_PORT env var.
"""

import json
import os
from urllib.parse import urlsplit

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.news_service.agent import news_agent

PORT = int(os.getenv("NEWS_AGENT_PORT", "8001"))

# What the agent card advertises as its own reachable address, kept separate
# from the bind address on purpose. uvicorn binds 0.0.0.0 so the service is
# reachable on every interface, but 0.0.0.0 is a bind wildcard rather than a
# dialable address - advertising it in the card would hand every client an
# unusable RPC URL.
#
# This is only the fall-back now: CardHostMiddleware below rewrites the served
# card per request, so this value is what a client sees only when its request
# carries no Host header at all. Set it when the service and its clients are on
# different hosts AND the middleware's Host-reflection is not wanted.
ADVERTISED_HOST = os.getenv("NEWS_AGENT_ADVERTISED_HOST", "localhost")

# Paths whose JSON body carries dialable addresses. The A2A specification fixes
# the first; the second is the pre-1.0 spelling, matched so an SDK upgrade that
# moves the card does not silently reintroduce the bug this middleware fixes.
_CARD_PATHS = frozenset(
    {"/.well-known/agent-card.json", "/.well-known/agent.json"}
)


class CardHostMiddleware(BaseHTTPMiddleware):
    """Serve the agent card with the address the client actually reached us on.

    ## The bug this fixes

    `to_a2a` bakes one address into the card at startup, and ADK's client
    (`RemoteA2aAgent`) dials whatever the card says rather than the URL it
    fetched the card from - `_compat.agent_card_url` reads
    `supportedInterfaces[i].url` (A2A 1.x) or the top-level `url` (0.3.x), and
    that is the address the RPC goes to. So a single baked-in address can only
    ever be right for one class of client.

    Under compose that bit. The card advertised `http://news-agent:8001`, which
    is correct for the agent container and unresolvable from the host, so every
    news question from a local checkout - `adk run`, an eval sweep, a locally
    run Streamlit UI - failed at DNS. Setting it to `localhost` instead just
    moves the failure: the agent container's localhost is the agent container.

    ## Why reflect the Host header rather than run two services

    An agent card is a self-description handed to a specific caller, and the
    address in it is only meaningful relative to that caller's network. The
    Host header is exactly the name that caller used to reach us, so echoing it
    back is the one answer that is correct for every caller at once - a host
    fetching via `localhost:8001` is told `localhost:8001`, the agent container
    fetching via `news-agent:8001` is told `news-agent:8001`.

    The alternative was to stop publishing the container's port and have local
    checkouts run their own News Agent process. Rejected on the precedent
    ADR-0020 already set for `mcp-fetch`: one transport, not two, because "the
    fallback path would be the one nobody exercises". A second code path that
    only local runs take is a second set of failure modes that only local runs
    discover.

    ## The limitation, stated rather than discovered later

    Reflecting a client-supplied header is right for a single-machine demo and
    wrong for a public deployment. `Host` is client-controlled, so a caller can
    make the card advertise any address it likes - harmless when the only thing
    it can do is mis-address its own next request, and not harmless if the card
    is ever cached and served to third parties, or if anything downstream
    treats the card's URL as trusted input. Behind a reverse proxy the same
    goes for `X-Forwarded-*`, which this deliberately does not read. If this
    service is ever exposed beyond localhost and a compose network, pin the
    address explicitly via NEWS_AGENT_ADVERTISED_HOST and take this middleware
    out.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        host = request.headers.get("host")
        if request.url.path not in _CARD_PATHS or not host:
            return response

        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            card = json.loads(body)
        except json.JSONDecodeError:
            # Not a card we understand (an error page, most likely). Pass it
            # through untouched rather than turning one failure into two.
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        _rewrite_card_addresses(card, netloc=host, scheme=request.url.scheme)
        return JSONResponse(content=card, status_code=response.status_code)


def _rewrite_card_addresses(card: dict, *, netloc: str, scheme: str) -> None:
    """Point every dialable address in `card` at `scheme://netloc`, in place.

    Only the fields the A2A spec defines as RPC endpoints are touched, named
    explicitly rather than found by walking the JSON for anything called "url".
    A card also carries documentation and provider URLs that belong to whoever
    published the agent, and rewriting those would be actively wrong - they are
    not addresses of this service.

    Both the 1.x spelling (`supportedInterfaces[].url`) and the 0.3.x one
    (top-level `url`) are handled, for the same reason `_CARD_PATHS` has two
    entries: the installed a2a-sdk emits the first, and ADK's own client still
    reads either, so an upgrade in the middle of that transition should not
    quietly leave the card unrewritten.
    """
    base = f"{scheme}://{netloc}"

    def repoint(url: str) -> str:
        # Keep the path (`to_a2a` builds an rpc_url ending in "/"); replace
        # only the part that identifies the host.
        return f"{base}{urlsplit(url).path or '/'}"

    if isinstance(card.get("url"), str):
        card["url"] = repoint(card["url"])
    for key in ("supportedInterfaces", "additionalInterfaces"):
        for interface in card.get(key) or []:
            if isinstance(interface, dict) and isinstance(interface.get("url"), str):
                interface["url"] = repoint(interface["url"])


# Module level so `uvicorn src.news_service.server:app` works as well as running
# this file directly - to_a2a returns a plain Starlette app either way.
app = to_a2a(news_agent, host=ADVERTISED_HOST, port=PORT, protocol="http")
app.add_middleware(CardHostMiddleware)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
