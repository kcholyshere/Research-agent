FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/ src/

ENV PATH="/app/.venv/bin:$PATH"

# ADK's dev UI by default; the optional Streamlit UI (src/ui/app.py) is an
# alternative entrypoint - swap the CMD below or run both containers. Verify
# in-container data access before relying on this (Finrag audit A3: the image
# ships no data/ or models/ - mount them).
EXPOSE 8000
CMD ["adk", "web", "src", "--host", "0.0.0.0"]
