# The MCP fetch server, reachable over the network.
#
# Phase 3 names Anthropic's reference `fetch` server and says to run its Docker
# image, so this image starts from that one and does not modify it. What it adds
# is a transport: `mcp/fetch` only speaks stdio (its serve() hardcodes
# stdio_server() and its CLI exposes no port), which means the only ways to
# reach it from another container are to mount the host Docker socket and spawn
# it per call, or to nest a Docker daemon. Both put container orchestration
# inside the agent process. Fronting the reference server with a stdio-to-HTTP
# proxy keeps it a plain compose service instead - see ADR-0020.
FROM mcp/fetch

# mcp-proxy's own dependency resolution pulls a newer `mcp` than the reference
# image ships, and in that version `request_ctx` is no longer importable from
# mcp.server.lowlevel.server, which is where the proxy imports it from - so
# installing mcp-proxy alone leaves it broken at import. Re-pinning `mcp` to the
# 1.23.0 the reference image was built against fixes the proxy and leaves the
# fetch server on the exact dependency it shipped with. Both versions are pinned
# because this pairing is the thing that works, not an arbitrary latest.
RUN pip install --no-cache-dir "mcp-proxy==0.12.0" "mcp==1.23.0"

EXPOSE 8090

# Server mode: run `mcp-server-fetch` as a stdio child and expose it over
# streamable HTTP at /mcp. --stateless because the client opens a fresh session
# per call and keeps no server-side state between them.
ENTRYPOINT ["mcp-proxy", "--host", "0.0.0.0", "--port", "8090", "--stateless", "mcp-server-fetch"]
