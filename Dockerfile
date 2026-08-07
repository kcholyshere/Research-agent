# The application image. One image, three roles: the ADK dev UI (default), the
# Streamlit chat UI, and the phase 5 News Agent service - they differ only by
# command, so compose overrides CMD rather than building three images with
# three different tags. Tagged once, as `research-agent-app` under
# docker-compose.yml's `x-app` anchor - see the comment there for the
# precise claim: compose still walks a build per service, but on this
# Dockerfile those are cache hits, and all three converge on the one tag.
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
# the property that makes the image reproducible. --no-dev drops the dev
# dependency group (ADR-0026): uv installs default groups unless told not to,
# and pytest has no business in a shipped image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/

# pyproject declares no build-system, so uv treats this as a virtual project:
# dependencies land in /app/.venv but `src` itself is imported from the working
# directory. That is why WORKDIR must stay /app for `python -m src....` to work.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Run as a non-root user (Finding 13): the agent processes content it does not
# control - web search results, an MCP-fetched page - and data/ is bind-mounted
# read-write, so a code-execution bug anywhere in that path currently runs as
# root against a host directory. UID/GID are build args rather than a fixed
# number because the two writable/readable bind mounts are host directories
# (./data - index build output and the phase 6 Canvas tool's artefacts under
# data/processed/artefacts; ./models, read-only) and Linux bind mounts enforce
# the real host permission bits: a container UID unrelated to the host-owning
# UID gets EACCES on write there, which would turn "runs as root" into "cannot
# write its own output". 1000 is the default because it is the first non-root
# UID on Debian/Ubuntu/Fedora, so the common case needs no override; a host
# whose checkout is owned by a different UID should set APP_UID/APP_GID (the
# APP_UID/APP_GID environment variables, via docker-compose.yml's x-app build
# args - see there) to `id -u`/`id -g` before building.
#
# Residual caveat, verified rather than assumed: on macOS Docker Desktop
# (virtiofs/gRPC-FUSE), a container UID with no relationship at all to the
# host-owning UID could still write into a host directory mode 755 owned by a
# different user, and read a host file mode 600 owned by a different user -
# tested directly against this checkout's own data/ and a copy of a gcloud
# credential file. Docker Desktop's bind-mount driver does not enforce the
# reported owner/permission bits the way a native Linux bind mount does, so
# the UID mismatch that matters on Linux is a non-issue here, which is the
# primary platform for this project. Do not take that as proof the APP_UID/
# APP_GID mechanism is unnecessary - it is what keeps the same image correct
# on a native Linux host, where the enforcement is real.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin app
# HOME is set explicitly rather than left to the NSS lookup of the passwd
# entry useradd just wrote - that lookup does resolve to /home/app correctly
# (verified: os.path.expanduser("~") as this user, with no HOME set, returns
# /home/app) - purely so anything that reads $HOME directly rather than going
# through getpwuid sees the same answer. This is also where the ADC bind mount
# now lands: docker-compose.yml mounts gcloud credentials to
# /home/app/.config/gcloud, not /root, because /root is mode 700 and a
# non-root user cannot even traverse into it - see the x-adc-mount comment
# there for what broke before that mount moved.
#
# Residual caveat found while verifying the agent and ui roles directly (both
# started, both served real traffic as this user): `adk web` logs one
# ERROR-level line at startup - "Failed to write runtime config file
# .../adk/cli/browser/assets/config/runtime-config.json: Permission denied" -
# because that path is under /app/.venv, owned by root from the `uv sync`
# step above. Checked against the installed google-adk source
# (cli/api_server.py): the write is wrapped in its own try/except IOError and
# only ever carries optional frontend branding (a custom logo), which this
# project does not set, so the dev UI still serves and behaves identically
# with the write failing - confirmed by loading it end-to-end as this user.
# Not fixed by chowning /app, because that would cost a layer over ~180
# packages for a write this project never needed to succeed in the first
# place; noted here so the ERROR line is not mistaken for a real regression
# the next time someone reads the logs.
ENV HOME=/home/app
USER app

# 8000 ADK dev UI, 8001 News Agent (A2A), 8501 Streamlit.
EXPOSE 8000 8001 8501

CMD ["adk", "web", "src", "--host", "0.0.0.0"]
