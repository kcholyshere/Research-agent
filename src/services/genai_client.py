from functools import lru_cache

from google import genai
from google.genai import types

from src import config

# Without a timeout, a dropped network connection (e.g. a wifi blip) leaves the
# SDK blocked on a dead socket indefinitely - observed hanging a whole eval run.
# 2 minutes comfortably covers the slowest legitimate generation call.
HTTP_TIMEOUT_MS = 120_000


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT,
        location=config.GCP_LOCATION,
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
    )
