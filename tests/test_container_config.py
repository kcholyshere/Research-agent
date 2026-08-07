"""Regression test for audit.md Findings 13 and 14c: no `USER` in the
Dockerfile, no `HEALTHCHECK` anywhere, and `depends_on` gating on mere
container start rather than readiness.

Finding 13's failure scenario is a race, not a crash: `docker compose up`
reports success the moment every container has *started*, which for
mcp-fetch or news-agent can be well before the process inside has bound its
port. A first question that reaches either service in that window sees
"server unreachable" even though compose said everything was fine. The fix
has two parts that only work together - a `HEALTHCHECK` that asks "is this
process actually ready", and a `depends_on: {condition: service_healthy}`
that makes the dependent services wait for that answer instead of for the
container object to merely exist.

Everything here is a static parse of docker-compose.yml and the Dockerfiles
it builds - no Docker daemon, no container, no network call. That split is
deliberate: "does the healthcheck's HTTP probe actually get a response" is a
question that needs a running container, so it belongs in an `integration`
test (or a manual `docker compose up` + `docker inspect ... .State.Health`,
which is how the probes here were originally verified); "is the invariant
that would make it pass even wired up at all" does not, and is what gates a
commit. Parsing rather than grepping for a literal string is what the audit
asked for directly: a compose reformat (key order, anchor renaming, flow vs
block style) must not fail this test, and a genuine regression - a
`depends_on` that reverts to a bare list, a deleted `healthcheck:` block, a
`USER` instruction that quietly comes back out - must.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE_PATH = _ROOT / "docker-compose.yml"
_DOCKERFILE_PATH = _ROOT / "Dockerfile"

_ROOT_USER_VALUES = {"root", "0", "0:0", "root:root"}


def _load_compose() -> dict[str, Any]:
    return yaml.safe_load(_COMPOSE_PATH.read_text())


def _services() -> dict[str, dict[str, Any]]:
    return _load_compose()["services"]


def _depends_on_targets(service: dict[str, Any]) -> dict[str, str | None]:
    """{target_service: condition} for one service's `depends_on`.

    Compose accepts a bare list (each entry implicitly `service_started`) or a
    mapping of service -> {condition: ...}. Both shapes are handled, rather
    than assuming the mapping form this file currently uses, precisely so a
    regression back to the list shorthand shows up as a real failure here
    instead of a silent parse mismatch.
    """
    depends_on = service.get("depends_on")
    if depends_on is None:
        return {}
    if isinstance(depends_on, list):
        return {name: None for name in depends_on}
    if isinstance(depends_on, dict):
        return {name: spec.get("condition") for name, spec in depends_on.items()}
    raise TypeError(f"Unexpected depends_on shape for a service: {depends_on!r}")


def _dockerfile_for(service: dict[str, Any]) -> Path:
    """Resolve which Dockerfile a compose service's `build` key points at.

    Handles the two shapes `build` can take (a bare context string, or a
    mapping with its own `context`/`dockerfile`), because the services in
    this file use both: mcp-fetch names `docker/mcp-fetch-bridge.Dockerfile`
    explicitly, while the shared `x-app` anchor relies on the default
    `Dockerfile` in its context.
    """
    build = service.get("build")
    if build is None or isinstance(build, str):
        context = build if isinstance(build, str) else "."
        return (_ROOT / context / "Dockerfile").resolve()
    context = build.get("context", ".")
    dockerfile = build.get("dockerfile", "Dockerfile")
    return (_ROOT / context / dockerfile).resolve()


def _parse_instructions(dockerfile_path: Path) -> list[tuple[str, str]]:
    """[(INSTRUCTION, rest-of-line)] for a Dockerfile.

    Not a full Dockerfile grammar - just enough to answer "does a HEALTHCHECK
    or USER instruction exist, and in what order relative to CMD/ENTRYPOINT",
    which is everything the tests below need. Comments and blank lines are
    dropped and backslash line-continuations are joined, so a HEALTHCHECK or
    USER instruction reformatted onto several lines is still recognised as
    one instruction rather than silently missed.
    """
    instructions: list[tuple[str, str]] = []
    pending = ""
    for raw_line in dockerfile_path.read_text().splitlines():
        line = raw_line.strip()
        if not pending and (not line or line.startswith("#")):
            continue
        pending = f"{pending}{line}" if pending else line
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip() + " "
            continue
        stripped = pending.strip()
        pending = ""
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        instructions.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
    return instructions


class TestDependsOnGatesOnReadiness:
    """Finding 13: nothing may depend on mere container start."""

    def test_no_depends_on_entry_defaults_to_service_started(self):
        offenders = [
            f"{name} -> {target} (condition={condition!r})"
            for name, service in _services().items()
            for target, condition in _depends_on_targets(service).items()
            if condition != "service_healthy"
        ]
        assert not offenders, (
            "depends_on entries not gated on service_healthy: "
            + ", ".join(offenders)
        )

    def test_every_service_depended_on_declares_a_health_check(self):
        services = _services()
        targets = {
            target
            for service in services.values()
            for target in _depends_on_targets(service)
        }
        missing = []
        for target in targets:
            service = services[target]
            healthcheck = service.get("healthcheck")
            if healthcheck is not None:
                # Declared directly on the compose service (news-agent's
                # case: it shares the main Dockerfile with two other CMD
                # roles that serve different ports, so a single baked-in
                # HEALTHCHECK can't fit all three - this is role-specific).
                # `disable: true` and `test: ["NONE"]` are compose's two
                # spellings for "no health check despite the key being
                # present" - either would make the service_healthy gating
                # above hang or fail at runtime, so both still count as
                # missing, and deliberately do NOT fall through to the
                # Dockerfile check below: an explicit compose-level disable
                # must not be silently rescued by an unrelated HEALTHCHECK
                # baked into the image it happens to build.
                if not healthcheck.get("disable") and healthcheck.get("test") != ["NONE"]:
                    continue
                missing.append(target)
                continue
            # No compose-level healthcheck key at all: fall back to the
            # Dockerfile the service actually builds (mcp-fetch's case - one
            # role, one Dockerfile, HEALTHCHECK baked in there instead).
            instructions = _parse_instructions(_dockerfile_for(service))
            if not any(instr == "HEALTHCHECK" for instr, _ in instructions):
                missing.append(target)
        assert not missing, (
            "services other services depend on with no health check, "
            f"in compose or their own Dockerfile: {missing}"
        )


class TestDockerfileDropsRoot:
    """Finding 13's other half: the image must not run as root."""

    def test_dockerfile_switches_to_a_non_root_user_before_its_entrypoint(self):
        instructions = _parse_instructions(_DOCKERFILE_PATH)

        user_indices = [i for i, (instr, _) in enumerate(instructions) if instr == "USER"]
        assert user_indices, "Dockerfile has no USER instruction"

        last_user_index = user_indices[-1]
        last_user_value = instructions[last_user_index][1].strip()
        assert last_user_value not in _ROOT_USER_VALUES, (
            f"final USER instruction switches to {last_user_value!r}, "
            "which is still root"
        )

        entrypoint_indices = [
            i for i, (instr, _) in enumerate(instructions) if instr in {"CMD", "ENTRYPOINT"}
        ]
        assert entrypoint_indices, "Dockerfile has no CMD/ENTRYPOINT"
        assert last_user_index < min(entrypoint_indices), (
            "USER instruction must come before CMD/ENTRYPOINT, not after"
        )


def test_the_user_facing_services_have_readiness_probes() -> None:
    """`docker compose up --wait` has to mean the stack is actually usable.

    Nothing depends on `agent` or `ui`, so these probes gate no other
    container and the health-check test above would not have caught their
    absence. They earn their place from a measured cold start: compose
    reports success the moment these containers start, and their ports are
    not bound for another 10-12 seconds while the ADK and google-genai import
    graph loads. A first-time user following the README gets
    connection-refused in that window - the same symptom as finding 13, from
    the top of the dependency chain rather than the bottom.
    """
    compose = _load_compose()
    for name in ("agent", "ui"):
        healthcheck = compose["services"][name].get("healthcheck")
        assert healthcheck, f"{name} has no readiness probe, so `up --wait` returns before it serves"
        assert healthcheck.get("test") not in (None, ["NONE"]), f"{name}'s probe is disabled"
