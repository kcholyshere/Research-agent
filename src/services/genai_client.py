from functools import lru_cache

from google import genai
from google.genai import types

from src import config

# Without a timeout, a dropped network connection (e.g. a wifi blip) leaves the
# SDK blocked on a dead socket indefinitely - observed hanging a whole eval run.
# 2 minutes comfortably covers the slowest legitimate generation call.
HTTP_TIMEOUT_MS = 120_000

# The same bound, for the agents' model calls - which do NOT go through the
# client above. Verified against the installed google-adk 2.5.0: ADK's Gemini
# wrapper builds its own genai.Client (models/google_llm.py, `api_client` is a
# cached_property) and sets no timeout, and google-genai's own default when
# http_options.timeout is None is float('inf'). So every agent model call in
# this project has been unbounded, while this module's HTTP_TIMEOUT_MS - which
# looks like it covers them - only ever applied to embedding calls.
#
# That is the 85-minute hang in TODOS: a Vertex/google_search call that never
# returned, 6s of CPU, no trace after the first minute. ADK's own `timeout`
# field on BaseAgent cannot fix it - it is inherited from the Workflow
# node runner and is silently ignored for a LoopAgent under InMemoryRunner.
#
# Applied per agent via generate_content_config.http_options, so it covers
# every entrypoint (adk run, adk web, Streamlit, the eval) rather than only
# the call sites that remember to pass a RunConfig. This bounds ONE model
# call, not a whole turn - a turn making N sequential calls can still take up
# to N times this. The turn-level bound stays the eval harness's
# asyncio.wait_for; production has no turn-level bound and this does not
# claim to add one.
MODEL_CALL_TIMEOUT_MS = 120_000


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT,
        location=config.GCP_LOCATION,
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
    )
