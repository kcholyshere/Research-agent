"""Regression test for audit.md finding 11: a model-calling agent added
after ADR-0013 without the model-call timeout the rule requires.

ADR-0013's rule is "every agent model call carries
genai_client.MODEL_CALL_TIMEOUT_MS via generate_content_config.http_options",
motivated by the 85-minute hang genai_client.py's docstring describes: ADK
builds its own genai.Client per agent and sets no timeout of its own, so an
agent missing this line has a model call with no ceiling at all.

The defect this test guards against is not "the news agent lacks a timeout" -
that instance is fixed by src/news_service/agent.py's http_options line. The
defect is that the rule was stated once (ADR-0013, when there were two
model-calling agents) and then silently violated by the THIRD and FOURTH
agents added later (web_search_agent, then news_agent - see audit.md finding
11), each only found by manual audit rather than by anything that runs.

A test built on a hand-maintained list of "these four agents" would repeat
the exact shape of the defect: ADR-0013 was a rule stated once in prose and
then silently outgrown; a hardcoded registry is the same rule stated once in
Python, just as capable of going stale the moment agent number five is added
elsewhere in the repo and nobody remembers to list it here. So this test has
two parts instead of one:

1. `test_registry_matches_every_agent_constructed_in_src` - an AST scan of
   every `.py` file under `src/` for a call whose function is named exactly
   `Agent` (bare or dotted, e.g. `agents.Agent(...)`), and asserts that set
   of (module, line) constructions matches _MODEL_CALLING_AGENTS exactly. A
   fifth `Agent(...)` added anywhere under src/ and left out of the registry
   fails THIS test, with the offending file:line in the message - that is
   what makes the registry self-checking rather than a second copy of the
   same trust-the-author problem ADR-0013 already failed once.

   Deliberately matched by exact name (`Agent`), not any *Agent* wrapper:
   `LoopAgent` (root_agent, agent.py), `RemoteA2aAgent`/
   `_ReachableRemoteA2aAgent` (news_remote_agent, news_agent.py) are
   different classes and do not call a model directly themselves - they wrap
   or transport an LlmAgent, they are not one. Excluding them by construct
   name rather than an explicit allowlist means a future rename can't
   silently smuggle a real model-calling agent past this scan under a
   plausible-looking new class name without the scan's own name check being
   the thing that has to change too.

2. `test_every_registered_agent_carries_the_model_call_timeout` - the actual
   ADR-0013 assertion, that every agent the registry (now proven complete by
   part 1) names has http_options.timeout == MODEL_CALL_TIMEOUT_MS.

Confirmed before writing this docstring, by grepping src/ for the same
pattern by hand: exactly four `Agent(` constructions exist under src/ today
(src/research_agent/agent.py, src/research_agent/critique.py,
src/tools/web_search.py, src/news_service/agent.py) - and, separately, that
src/evaluation/ab_harness.py's "compares agent variants" only ever takes
already-built `BaseAgent` instances as arguments; it constructs none of its
own, so it is correctly outside this test's scope rather than a fifth
instance missed by the scan.

Entirely offline: no Vertex call, no credentials, no network. The AST scan
only reads source text. Importing the four agent modules (in part 2) needs
no credentials either - not because anything here asserts that, but because
genai_client.get_client() (the only call in this project's import graph that
would need Application Default Credentials) is lru_cache-decorated and
never invoked at module scope in any of the four - confirmed by reading
src/services/genai_client.py directly, not inferred from the imports
succeeding.
"""

from __future__ import annotations

import ast
from pathlib import Path

from google.adk.agents import Agent

from src import config
from src.services.genai_client import MODEL_CALL_TIMEOUT_MS

_SRC_ROOT = config.PROJECT_ROOT / "src"

# Every model-calling agent this project defines, as (module path, attribute
# path) pairs, keyed by the source file the Agent(...) call lives in (used
# to cross-check against the AST scan below). "Attribute path" rather than a
# bare attribute name because web_search_agent is private (module-internal,
# wrapped in an AgentTool) and is only reachable via web_search_tool.agent -
# see src/tools/web_search.py.
_MODEL_CALLING_AGENTS: dict[str, tuple[str, str]] = {
    "src/research_agent/agent.py": ("src.research_agent.agent", "research_agent"),
    "src/research_agent/critique.py": ("src.research_agent.critique", "critique_agent"),
    "src/tools/web_search.py": ("src.tools.web_search", "web_search_tool.agent"),
    "src/news_service/agent.py": ("src.news_service.agent", "news_agent"),
}


def _resolve(module_path: str, attribute_path: str) -> Agent:
    """Import `module_path` and walk `attribute_path` (dotted) off it."""
    import importlib

    obj = importlib.import_module(module_path)
    for part in attribute_path.split("."):
        obj = getattr(obj, part)
    return obj


def _files_constructing_agent(src_root: Path) -> dict[str, list[int]]:
    """AST-scan every .py file under `src_root` for `Agent(...)` calls.

    Matches only the exact name `Agent`, as a bare name (`Agent(...)`) or as
    the final attribute of a dotted access (`agents.Agent(...)`) - not
    `LlmAgent`, `LoopAgent`, `RemoteA2aAgent` or anything else with "Agent"
    in it, since those wrap or transport an agent rather than calling a
    model directly (see module docstring for why that line is drawn here).
    Returns {relative_path: [line numbers]} for every match, so a file with
    more than one construction (there are none today) is still fully
    reported rather than silently deduped.
    """
    hits: dict[str, list[int]] = {}
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == "Agent":
                rel = str(path.relative_to(src_root.parent))
                hits.setdefault(rel, []).append(node.lineno)
    return hits


def test_registry_matches_every_agent_constructed_in_src() -> None:
    """The registry above must name every `Agent(...)` construction under src/.

    This is the test that keeps _MODEL_CALLING_AGENTS honest: without it, a
    fifth model-calling agent added in a new file - or a sixth added to an
    existing one - would simply not be in the registry, and
    test_every_registered_agent_carries_the_model_call_timeout would keep
    passing having never looked at it, exactly reproducing finding 11 (a
    rule stated once and then silently outgrown) one level down.
    """
    found = _files_constructing_agent(_SRC_ROOT)
    found_files = set(found)
    registered_files = set(_MODEL_CALLING_AGENTS)

    missing_from_registry = found_files - registered_files
    stale_in_registry = registered_files - found_files

    assert not missing_from_registry, (
        "Agent(...) is constructed in these files but none of them is in "
        f"tests/test_model_call_bounds.py's _MODEL_CALLING_AGENTS registry, so their "
        f"model-call timeout goes unchecked: { {f: found[f] for f in missing_from_registry} }. "
        "Add each one to the registry (and to _MODEL_CALLING_AGENTS's module/attribute pair)."
    )
    assert not stale_in_registry, (
        "the registry names these files as constructing an Agent, but the AST scan "
        f"found none there any more (renamed, removed, or refactored?): {stale_in_registry}. "
        "Update or remove the stale registry entries."
    )


def test_every_registered_agent_carries_the_model_call_timeout() -> None:
    """The actual ADR-0013 assertion: http_options.timeout == MODEL_CALL_TIMEOUT_MS.

    This is finding 11 reproduced as a check rather than as a manual read of
    four files: news_agent (src/news_service/agent.py) was added after
    research_agent and critique_agent already carried this bound, and did
    not carry it. Run against the pre-fix code (http_options omitted from
    news_agent's GenerateContentConfig), this test fails on exactly that
    agent - confirmed manually while writing this test, by reverting the
    fix, before restoring it.
    """
    missing: list[str] = []
    wrong_value: list[str] = []

    for module_path, attribute_path in _MODEL_CALLING_AGENTS.values():
        agent = _resolve(module_path, attribute_path)
        http_options = agent.generate_content_config.http_options
        if http_options is None or http_options.timeout is None:
            missing.append(f"{module_path}.{attribute_path} ({agent.name})")
        elif http_options.timeout != MODEL_CALL_TIMEOUT_MS:
            wrong_value.append(
                f"{module_path}.{attribute_path} ({agent.name}): "
                f"timeout={http_options.timeout}, expected {MODEL_CALL_TIMEOUT_MS}"
            )

    assert not missing, (
        "these model-calling agents have no http_options.timeout at all, so "
        f"their model calls are unbounded (see genai_client.MODEL_CALL_TIMEOUT_MS): {missing}"
    )
    assert not wrong_value, (
        "these agents set http_options.timeout to something other than the "
        f"shared MODEL_CALL_TIMEOUT_MS constant, which reintroduces the drift "
        f"ADR-0013 exists to avoid: {wrong_value}"
    )
