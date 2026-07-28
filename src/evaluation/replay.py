"""Record/replay fixtures for the agent's tools - what makes routing and
synthesis assertions deterministic, fast and free.

Why this exists at all: web and market answers change daily, so static
ground truth for them is impossible. Identical code and an identical
question gave 1 web search and 45.5s one day, and 5 searches and roughly
100s the next, purely because "yesterday" moved past the last real search
match (references/evaluation_brainstorm.md). Without record/replay, a run
that changes shape can never be told apart from a run of unchanged code
against a world that simply moved on - every comparison across days would be
noise. Recording real tool outputs once and replaying them pins the world so
the only thing left to vary between two runs is the code under test.

Where this intercepts: at the tool-function boundary, not the model or the
Runner. Recording/replaying whole agent turns would also freeze the model's
own behaviour, which is exactly the thing a routing or synthesis regression
test needs to still exercise for real. Freezing only the tools' outputs
keeps the LLM's planning, tool selection and synthesis live against fixed
inputs.

## The two interception points, verified against the installed google-adk
2.5.0 package (not docs - CLAUDE.md is emphatic that this API has been wrong
often enough to cost real time, and it very nearly did here too):

1. Plain function tools (`search_documents`, `get_financial_data`, both
   passed via `tools=[...]` on the `Agent`) are NOT wrapped into a
   `FunctionTool` once at construction time. `LlmAgent.canonical_tools()`
   calls `_convert_tool_union_to_tools()` on every invocation, and for a
   plain callable that does `FunctionTool(func=tool_union)` fresh each time,
   reading straight out of the agent's own `tools` list. A throwaway probe
   confirmed two fresh `canonical_tools()` calls against the same `Agent`
   return two different `FunctionTool` objects wrapping the *same* function -
   so patching the module-level function (or any previously-returned
   `FunctionTool`) does nothing; the only thing that sticks is mutating the
   entry in `agent.tools` itself, in place, before any call happens. That
   list is a mutable Python list even though `tools` is a pydantic field on
   `LlmAgent` - index assignment on it doesn't go through pydantic's
   attribute-assignment validation, only `agent.tools = ...` would.

2. `web_search_tool` is an `AgentTool` wrapping a sub-agent
   (src/tools/web_search.py), and `AgentTool` is already a `BaseTool`
   instance - `_convert_tool_union_to_tools()` returns it unchanged
   (`isinstance(tool_union, BaseTool): return [tool_union]`), so the same
   object comes back from every `canonical_tools()` call. That makes it safe
   to monkeypatch its bound `run_async` method directly on the instance and
   restore the original bound method afterwards - unlike the plain
   functions, there is no fresh wrapper to chase.

   One more thing the probe turned up, worth flagging rather than quietly
   working around: `AgentTool.__init__` names the tool after the wrapped
   sub-agent (`super().__init__(name=agent.name, ...)`), so the tool the LLM
   actually calls is named "web_search_agent", not "web_search_tool" - the
   Python variable name in src/tools/web_search.py. `schema.py`'s
   `TOOL_TO_ROUTE` dict keys on "web_search_tool", which will never match a
   real event's `call.name`. That is a latent bug in a file this module is
   told not to touch; this module keys its own fixtures on the name ADK
   actually uses (`tool.name`, i.e. "web_search_agent"), which is correct
   for its own purpose regardless of that mismatch elsewhere.

3. Invocation is async throughout in this version - `FunctionTool.run_async`
   calls `_invoke_callable`, which awaits the target if
   `inspect.iscoroutinefunction(target)` is true and calls it directly
   otherwise, checking the *actual object invoked* (our wrapper), not the
   original function. Wrapping both `search_documents` (sync) and
   `get_financial_data` (async) in one `async def wrapper` is therefore safe
   either way - it is always treated as async, and it awaits or calls the
   real function underneath as appropriate.

## What deliberately is NOT patched

Sub-agents are discovered recursively (`root_agent` is a `LoopAgent` with no
`tools` of its own - the tools live on its `research_agent` sub-agent), which
means `critique_agent` and its `exit_loop` tool are reachable by the same
walk. `exit_loop` is excluded on purpose, via `schema.py`'s own
`CONTROL_TOOLS` set (reused rather than re-declared, so the two files can't
drift apart on what counts as a control call). Patching it the same way as
an evidence tool would be a real bug, not just noise: `exit_loop`'s job is a
side effect on the *live* `ToolContext` (`tool_context.actions.escalate =
True`), which is what tells `LoopAgent` to stop. A replay that served a
cached return value instead of calling the real function would never set
that flag, and the loop would run to `max_iterations` on every replayed turn
regardless of what the critique agent actually decided - a much worse
failure than the noisy fixture entries excluding it avoids.

## Fixture file format

A single JSON array of `{"tool": str, "args": {...}, "response": ...}"`
objects, written with stable key ordering (`sort_keys=True`) and indentation
so a reviewer sees exactly one new hunk per recorded call in a diff - not a
wholesale reformat of the file. Record mode loads whatever is already on
disk and appends to it (rather than starting fresh each run), so recording
one new question's fixtures does not clobber every other question's.

Lookup key: `(tool name, normalised arguments)`, where normalisation is
`json.dumps(args, sort_keys=True)` - recursive by construction, so nested
dict argument values are order-independent too, not just the top level.
Distinct arguments to the same tool therefore get distinct entries and
replay correctly, per call, within one turn. Calls that recorded the exact
same (tool, arguments) key more than once - e.g. two different record
sessions, or a genuinely repeated call within one turn - are served back in
the order they were recorded (a FIFO queue per key), so a second identical
call does not just keep replaying the first call's answer forever.

## Miss handling

A replay lookup that finds nothing is never silently papered over and never
falls through to a live call - both would let a real behavioural change (the
agent starting to call a tool with different arguments than the fixture
expects) masquerade as a pass. A miss is recorded in
`FixtureSession.misses` as a human-readable description, and the call
returns a clearly-marked sentinel payload so the rest of the turn - and
whatever routing data it produces - still completes and is still worth
capturing.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.tools.agent_tool import AgentTool

from src.evaluation.schema import CONTROL_TOOLS

FixtureMode = Literal["live", "record", "replay"]

_MISS_TAG = "REPLAY_FIXTURE_MISS"


@dataclass
class FixtureSession:
    """Handle returned by `tool_fixtures` for the duration of its `with` block.

    `calls_recorded` counts fixture entries actually written to disk this
    session - so it is necessarily 0 in "replay" and "live" modes. It is not
    overloaded to also mean "calls served from an existing fixture" in
    replay mode: that guarantee (no real tool call happens) is structural -
    `_FixtureStore.handle` never invokes the real tool in replay mode at all -
    and doesn't need a counter to prove it. `misses` is the number that
    matters for replay: it is empty exactly when every call this turn made
    was covered by the fixture file.
    """

    mode: FixtureMode
    misses: list[str] = field(default_factory=list)
    calls_recorded: int = 0


def _normalise_args(args: dict[str, Any]) -> str:
    """Canonical string key for a call's arguments.

    `sort_keys=True` makes key order irrelevant recursively (json.dumps
    applies it to nested dicts too), which matters because the LLM does not
    guarantee argument order is stable across otherwise-identical calls.
    `default=str` is a defensive fallback only - every tool here is called
    with plain JSON-safe arguments (strings), so it should never trigger.
    """
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _load_fixture_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return json.loads(raw)


def _function_miss_sentinel(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    # A dict, matching the shape every plain-function tool here already
    # returns (search_documents: list[dict], get_financial_data: dict) well
    # enough that it serialises the same way into the model's context - the
    # LLM reads it as data either way, and the explicit tag is what stops a
    # miss from reading like an empty-but-real result.
    return {
        "_replay_status": _MISS_TAG,
        "detail": (
            f"No recorded fixture for {tool_name} with args {args!r}. Replay "
            "never calls the real tool, so this call has no real evidence "
            "behind it - see FixtureSession.misses."
        ),
    }


def _agent_tool_miss_sentinel(tool_name: str, args: dict[str, Any]) -> str:
    # A plain string, matching what AgentTool.run_async returns for real
    # (the wrapped sub-agent's merged final text) - so a miss looks like
    # unusual prose to the calling agent rather than an unexpected type.
    return (
        f"[{_MISS_TAG}] No recorded fixture for {tool_name} with args "
        f"{args!r}. Replay never calls the real sub-agent, so this call has "
        "no real evidence behind it - see FixtureSession.misses."
    )


class _FixtureStore:
    """Backs every patched tool for one `tool_fixtures` session.

    One store is shared by every wrapper created for a given `with` block, so
    a single fixture file and a single `FixtureSession` (misses,
    calls_recorded) stay consistent across however many tools got patched.
    """

    def __init__(self, mode: Literal["record", "replay"], path: Path, session: FixtureSession) -> None:
        self._mode = mode
        self._path = path
        self._session = session
        self._entries: list[dict[str, Any]] = _load_fixture_file(path)

        # Built once at entry, not per-call: replay must never touch disk or
        # re-derive state mid-turn, and a dict-of-deques gives O(1) lookup
        # plus first-recorded-first-replayed order for repeated identical
        # calls.
        self._replay_index: dict[tuple[str, str], deque[Any]] = {}
        if mode == "replay":
            for entry in self._entries:
                key = (entry["tool"], _normalise_args(entry["args"]))
                self._replay_index.setdefault(key, deque()).append(entry["response"])

    async def handle(
        self,
        tool_name: str,
        args: dict[str, Any],
        real_call: Callable[[], Any],
        miss_sentinel: Any,
    ) -> Any:
        if self._mode == "record":
            response = await real_call()
            self._entries.append({"tool": tool_name, "args": args, "response": response})
            self._flush()
            self._session.calls_recorded += 1
            return response

        # Replay: real_call is never invoked, by construction - this branch
        # is the entire guarantee that replay makes no network or model call.
        key = (tool_name, _normalise_args(args))
        queue = self._replay_index.get(key)
        if queue:
            return queue.popleft()
        self._session.misses.append(f"{tool_name} args={args!r}")
        return miss_sentinel

    def _flush(self) -> None:
        # Rewritten whole-file per call rather than batched at exit: a run
        # that hangs and gets killed by an external timeout (ab_harness has
        # no built-in bound today - see the evaluation brainstorm's "nothing
        # bounds a turn's duration" trap) should still keep whatever it
        # recorded before the hang, not lose the entire session.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _iter_llm_agents(agent: BaseAgent) -> Iterator[LlmAgent]:
    """Walk `agent` and its sub-agents, yielding every `LlmAgent` found.

    Needed because the object handed to `tool_fixtures` is not always the
    tool-bearing agent itself: `root_agent` (src/research_agent/agent.py) is
    a `LoopAgent` wrapping `research_agent`, and `LoopAgent` has no `tools`
    attribute of its own - only its `LlmAgent` sub-agents do. Recursing once
    handles both today's `root_agent` and a deeper composition later without
    the caller having to know which layer actually holds the tools.
    """
    if isinstance(agent, LlmAgent):
        yield agent
    for sub_agent in getattr(agent, "sub_agents", None) or []:
        yield from _iter_llm_agents(sub_agent)


def _make_function_wrapper(
    tool_name: str, original: Callable[..., Any], store: _FixtureStore
) -> Callable[..., Any]:
    """Build the record/replay stand-in for one plain-function tool.

    `functools.wraps` is load-bearing, not cosmetic: `FunctionTool.__init__`
    reads `func.__name__` for the tool's name, and `build_function_declaration`
    reads `inspect.signature(func)` - which follows `__wrapped__` by default -
    for the parameter schema the LLM plans against. Losing either would
    change what the LLM sees, not just how the call gets served underneath.
    """
    is_async = inspect.iscoroutinefunction(original)

    @functools.wraps(original)
    async def wrapper(**kwargs: Any) -> Any:
        async def real_call() -> Any:
            return await original(**kwargs) if is_async else original(**kwargs)

        return await store.handle(
            tool_name, kwargs, real_call, _function_miss_sentinel(tool_name, kwargs)
        )

    return wrapper


def _make_agent_tool_wrapper(
    tool_name: str, original_run_async: Callable[..., Any], store: _FixtureStore
) -> Callable[..., Any]:
    """Build the record/replay stand-in for one `AgentTool`'s `run_async`.

    Matches `AgentTool.run_async`'s own signature (keyword-only `args` and
    `tool_context`) exactly, since it replaces the bound method directly
    rather than going through any re-wrapping step - unlike the plain
    function tools, ADK never rebuilds this object.
    """

    async def wrapper(*, args: dict[str, Any], tool_context: Any) -> Any:
        async def real_call() -> Any:
            return await original_run_async(args=args, tool_context=tool_context)

        return await store.handle(
            tool_name, args, real_call, _agent_tool_miss_sentinel(tool_name, args)
        )

    return wrapper


def _patch_tools(agent: BaseAgent, store: _FixtureStore) -> list[Callable[[], None]]:
    """Patch every evidence-gathering tool found under `agent`; return restorers.

    Each restorer is a zero-argument callable that undoes exactly one patch;
    `tool_fixtures` runs all of them in `finally` regardless of how the `with`
    block exits.
    """
    restorers: list[Callable[[], None]] = []

    for llm_agent in _iter_llm_agents(agent):
        tools = llm_agent.tools  # the live list object - mutate in place, see module docstring point 1
        for index, tool in enumerate(tools):
            if isinstance(tool, AgentTool):
                original_run_async = tool.run_async
                tool.run_async = _make_agent_tool_wrapper(tool.name, original_run_async, store)
                restorers.append(functools.partial(setattr, tool, "run_async", original_run_async))
            elif callable(tool):
                name = getattr(tool, "__name__", None)
                # See module docstring "What deliberately is NOT patched":
                # exit_loop's real side effect on the live ToolContext must
                # keep running every time, in every mode - a cached replay of
                # its return value would never set actions.escalate, and the
                # loop would silently stop honouring the critique agent.
                if name is None or name in CONTROL_TOOLS:
                    continue
                original = tool
                tools[index] = _make_function_wrapper(name, original, store)
                restorers.append(functools.partial(tools.__setitem__, index, original))
            # Anything else (a BaseToolset, say) is left untouched - nothing
            # in this project currently attaches one, and silently patching a
            # tool type this module hasn't verified against the installed
            # package would violate the same rule that motivated verifying
            # the two cases above in the first place.

    return restorers


@contextmanager
def tool_fixtures(agent: BaseAgent, mode: FixtureMode, fixture_path: Path) -> Iterator[FixtureSession]:
    """Patch `agent`'s tools for the duration of the block, then restore them exactly.

    `mode="live"` takes no action at all - no discovery, no patching, no file
    I/O - so it can never distort a live latency measurement, which is the
    other thing this same fixture layer needs to stay honest for (see
    references/evaluation_brainstorm.md's replay-mode section: live mode is
    kept specifically for latency and operational health, separate from the
    deterministic content assertions replay mode exists for).
    """
    session = FixtureSession(mode=mode, misses=[], calls_recorded=0)

    if mode == "live":
        yield session
        return

    store = _FixtureStore(mode=mode, path=fixture_path, session=session)
    restorers = _patch_tools(agent, store)
    try:
        yield session
    finally:
        # Always run every restorer, even if the block above raised - a
        # leaked patch would silently corrupt every later run in the same
        # process (the same class of bug the ab_harness session-reuse trap
        # describes, one layer down).
        for restore in restorers:
            restore()
