"""Makes an `AgentTool` carry the same `fact` argument the plain function tools do.

## Why this exists

ADR-0024's declared-plan gate was supposed to enforce that one named source
is authoritative for one named fact. The 2026-08-06 audit (finding 3) found
it enforced something weaker: `record_declared_plan` kept a flat set of tool
names and discarded `facts` entirely, so on a plan with more than one fact
every declared tool was legal for every fact. The audit's own example: a plan
declaring `get_financial_data` for "BTC price" and `web_search_agent` for
"background on the ETF approval" would happily let the price be answered from
the web, because the web tool was in the declared set.

Fixing that needs the gate to know WHICH fact a given call is serving, and
that signal does not exist. Verified against the installed `google-adk`
2.5.0: `before_tool_callback` is invoked as `callback(tool=, args=,
tool_context=)` (`flows/llm_flows/functions.py`), and `_create_tool_context`
threads through only `function_call.id`, not the function call itself, not
its containing `Content`, and not the narration around it. `ToolContext` and
`CallbackContext` are the same `Context` class and carry no field that could
stand in either. Gemini also batches parallel calls into one `Content`, so
even reading the surrounding text could not attribute a sentence to one call
among several.

So the binding has to be created rather than read: every evidence tool takes
a `fact` argument naming the declared fact it is serving, and
`tool_budget.enforce_tool_budget` checks that fact's declared source against
the tool being called. `report_gap(fact, source_checked)` already established
this convention in this project, which is why the argument is named `fact`
here too rather than something new.

## Why a subclass rather than a wrapper function

`search_documents` and `get_financial_data` are plain functions, so they just
grow a parameter. The web search and News Agent tools are not: both are
`AgentTool`s, and `AgentTool` builds its own declaration. With no
`input_schema` on the wrapped agent - and neither has one, the News Agent's
`RemoteA2aAgent` not even being an `LlmAgent` - `_get_declaration` emits a
fixed `{"request": string}` schema and `run_async` reads `args["request"]`.
There is no hook to add a parameter.

Wrapping each in a plain function tool instead was rejected: `AgentTool`
names itself after the agent it wraps, so a function wrapper would change
`tool.name` from `web_search_agent`/`news_agent` to whatever the function is
called, and that name is load-bearing in at least four places -
`schema.TOOL_TO_ROUTE`, `declare_plan.EVIDENCE_TOOL_NAMES`,
`tool_budget.OUTPUT_TOOLS`'s sibling checks, and the eval's stored fixtures.
Subclassing keeps the name identical and changes only the declaration.

The subclass delegates to `super()` in both methods rather than
reimplementing them. `_get_declaration` has two branches in the installed
version, gated on a feature flag, emitting `parameters_json_schema` in one
and a `types.Schema` in the other; copying either would silently diverge the
moment that flag flips. Injecting into whichever shape came back cannot.

## `fact` is stripped before the sub-agent sees it

`run_async` pops `fact` before delegating. It is addressed to the gate, not
to the wrapped agent, and leaving it in would change what the sub-agent is
asked: `AgentTool.run_async` falls back to `json.dumps(args)` as the prompt
when `request` is absent, and even with `request` present an extra key is
noise in a payload this project has already tuned. Popping also keeps the
eval's replay fixtures keyed on the arguments the sub-agent actually
received.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# The argument name every evidence tool uses, and the key
# `tool_budget.enforce_tool_budget` reads out of `args`. Defined here rather
# than in tool_budget.py because this module is what puts it into the two
# AgentTool declarations - the two plain function tools spell it in their own
# signatures, where a constant could not reach.
FACT_ARG = "fact"

# What the model is told the argument is for. This is a tool description, so
# it is load-bearing rather than decoration: it is the only place the planner
# learns that the value has to match a declared fact rather than being free
# text. Kept identical to the wording in the two plain tools' docstrings, so
# the model sees one rule and not four variants of it.
FACT_ARG_DESCRIPTION = (
    "The fact from your declared plan (declare_plan) that this call is gathering, "
    "copied exactly as you wrote it there. The declared source for that fact must "
    "be this tool, or the call is refused."
)


class FactTaggedAgentTool(AgentTool):
    """An `AgentTool` whose declaration also requires a `fact` argument.

    See this module's docstring for why the binding has to be created rather
    than read from the callback context, and why this is a subclass rather
    than a function wrapper.
    """

    def _get_declaration(self) -> types.FunctionDeclaration:
        declaration = super()._get_declaration()

        # Two shapes, because the installed AgentTool emits one or the other
        # depending on the JSON_SCHEMA_FOR_FUNC_DECL feature flag. Both are
        # handled by mutating what super() returned rather than by rebuilding
        # it, so a flag flip cannot leave this branch behind.
        schema = declaration.parameters_json_schema
        if isinstance(schema, dict):
            properties = schema.setdefault("properties", {})
            properties[FACT_ARG] = {"type": "string", "description": FACT_ARG_DESCRIPTION}
            required = schema.setdefault("required", [])
            if FACT_ARG not in required:
                required.append(FACT_ARG)
            return declaration

        if declaration.parameters is not None:
            declaration.parameters.properties[FACT_ARG] = types.Schema(
                type=types.Type.STRING, description=FACT_ARG_DESCRIPTION
            )
            required = declaration.parameters.required or []
            if FACT_ARG not in required:
                declaration.parameters.required = [*required, FACT_ARG]
            return declaration

        # No parameters at all is not a shape the installed version produces
        # for either of this project's two AgentTools, but failing loudly
        # beats emitting a declaration the gate can never match - a tool the
        # model can call without a fact is a hole in the gate, and a silent
        # one.
        raise ValueError(
            f"{self.name}'s declaration carries neither parameters_json_schema nor "
            "parameters, so the fact argument cannot be added to it. The gate in "
            "tool_budget.enforce_tool_budget would refuse every call to this tool."
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        """Strip `fact` before the wrapped agent runs.

        By the time this executes the gate has already read `fact` out of the
        same dict in `before_tool_callback` and decided the call is allowed,
        so removing it here costs nothing and keeps the sub-agent's prompt
        exactly what it was before fact tagging existed.
        """
        forwarded = {key: value for key, value in args.items() if key != FACT_ARG}
        return await super().run_async(args=forwarded, tool_context=tool_context)
