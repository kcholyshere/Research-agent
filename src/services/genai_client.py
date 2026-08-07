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


# Audit finding 15a: config.GCP_PROJECT has no default and no validation at
# the point it is read (src/config.py), and that is deliberate - config.py is
# imported unconditionally by modules that never touch Vertex, and the whole
# offline test suite (tests/conftest.py's ToolContext fixtures included)
# imports project modules with no credentials present. A hard failure at
# config-import time would break collecting that suite on any machine or CI
# box without a .env, for tests that never need Vertex at all.
#
# This function is the real gate instead: it is the one place a genai.Client
# actually gets constructed for our own code (embedder.py's embed calls,
# langfuse_sync.py), so the check below only runs for callers who are about
# to make a real Vertex call, and only then. Verified against the installed
# google-genai (_api_client.py): BaseApiClient reads env_project =
# os.environ.get('GOOGLE_CLOUD_PROJECT'), then self.project = project or
# env_project, and if that is still falsy, load_auth() resolves a project
# from Application Default Credentials instead of raising - so a blank
# GOOGLE_CLOUD_PROJECT does not fail, it silently sends every call to
# whichever project ADC defaults to. That is the exact misdiagnosis ADR-0004
# spent real time on: a 404 that reads as "model retired from the catalogue"
# when the actual cause was "wrong project". The check below turns a blank
# project into a named, loud error before genai.Client is ever built, instead
# of a working client pointed at the wrong place.
#
# Residual gap, not closed by this check: ADK's own agent model calls do not
# go through this function at all (see MODEL_CALL_TIMEOUT_MS above) - ADK
# builds its own genai.Client internally and reads GOOGLE_CLOUD_PROJECT from
# the environment directly. So a blank project still lets a turn's LLM calls
# through to ADC's default project even after this fix; this only gates the
# calls that route through get_client() (embeddings today).
#
# lru_cache does not cache a raised exception - only a successful return - so
# fixing the environment and retrying (e.g. between test cases, or after
# editing .env) is not blocked by an earlier failed call.
@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    if not config.GCP_PROJECT:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set (src.config.GCP_PROJECT, read "
            "from the environment/.env by src/config.py, is empty). Without "
            "it, google.genai.Client falls back to Application Default "
            "Credentials' own default project instead of failing - every "
            "Vertex call would silently go to whichever project ADC "
            "defaults to, not the one you intended. Set GOOGLE_CLOUD_PROJECT "
            "in your .env file (see .env.example) or the environment before "
            "calling get_client()."
        )
    return genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT,
        location=config.GCP_LOCATION,
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
    )
