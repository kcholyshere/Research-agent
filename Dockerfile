# The application image. One image, three roles: the ADK dev UI (default), the
# Streamlit chat UI, and the phase 5 News Agent service - they differ only by
# command, so compose overrides CMD rather than building three images.
#
# Deliberately ships no corpus and no FAISS index. `models/` is gitignored and
# the index is a build product of `python -m src.dataset`, so baking it in would
# mean an embedding pass (and Vertex credentials) at build time and an image
# that goes stale the moment the corpus changes. compose bind-mounts both
# instead - see ADR-0020. The consequence to know: without those mounts the
# knowledge base is empty and document search fails at first query, not at
# startup.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependencies before source, so editing src/ does not invalidate the layer that
# resolves and installs ~180 packages. --frozen fails the build if uv.lock and
# pyproject.toml have drifted apart rather than silently re-resolving, which is
# the property that makes the image reproducible.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/ src/

# pyproject declares no build-system, so uv treats this as a virtual project:
# dependencies land in /app/.venv but `src` itself is imported from the working
# directory. That is why WORKDIR must stay /app for `python -m src....` to work.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# 8000 ADK dev UI, 8001 News Agent (A2A), 8501 Streamlit.
EXPOSE 8000 8001 8501

CMD ["adk", "web", "src", "--host", "0.0.0.0"]
