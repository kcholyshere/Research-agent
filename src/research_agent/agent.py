"""The core research agent: a Plan-Execute-Synthesize flow over four evidence
tools - the private-knowledge-base Document Search Tool (phase 1), the
public-internet Web Search Tool (phase 2), the Financial Data Tool (phase 3),
and the A2A News Agent delegation (phase 5) - plus the phase 6 Canvas output
tool, all wrapped in a phase 4 critique/refinement loop.

ADK conventions: this module exposes `root_agent`, which `adk run src/research_agent`
and `adk web src` discover by name. That symbol is now the `LoopAgent`
(see ADR-0010), not the single-shot planner directly - `research_agent`
below is the renamed former `root_agent`, one sub-agent of the loop.
Plain functions passed via `tools=` are auto-wrapped as function tools, with
their docstrings as tool descriptions; `web_search_tool` is an `AgentTool`
wrapping a sub-agent (see src/tools/web_search.py for why).

Observability: Langfuse tracing is wired in here, not per-entrypoint, since
every entrypoint (`adk run`, `adk web`, the Streamlit UI) imports this module
to get `root_agent` - one instrumentation call covers all three. It must run
before any Agent is constructed (including web_search.py's module-level
sub-agent, and critique.py's module-level `critique_agent`), and after
`src.config` has loaded `.env`, hence the import order below.

Repo-root sys.path bootstrap: ADK's CLI loader (`adk run`/`adk web`) inserts
only the agent's *parent* directory (`src`) onto sys.path, not the repo root,
so a bare `from src import config` fails under that entrypoint even though it
works under `python -m` and Streamlit (both of which put the repo root on
sys.path via cwd). Fixing it up here, from this file's own location, makes
`from src import config` work under every entrypoint without relying on the
caller to invoke things a particular way.
"""

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src import config  # noqa: F401 - import first: triggers .env load via dotenv

from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from langfuse import get_client

GoogleADKInstrumentor().instrument()
langfuse_client = get_client()

from typing import Any

from google.adk.agents import Agent, LoopAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from src.services import genai_client
from src.tools.canvas import create_canvas
from src.tools.declare_plan import declare_plan
from src.tools.document_search import search_documents
from src.tools.financial_data import get_financial_data
from src.tools.news_agent import news_agent_tool
from src.tools.report_gap import report_gap
from src.tools.web_search import web_search_tool

# Imported after instrument() (see the observability note above) - this
# module's own import constructs critique_agent = Agent(...) at load time.
from src.research_agent import token_budget, tool_budget
from src.research_agent.critique import LAST_ARTEFACT_KEY, critique_agent, reset_turn_state


def _after_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """Dispatch on tool name to the two callbacks that need to see a tool's own response.

    Signature verified against the installed google-adk 2.5.0 rather than taken
    from docs, per CLAUDE.md: `AfterToolCallback` is
    `Callable[[BaseTool, dict[str, Any], ToolContext, dict], Optional[dict]]`,
    and the flow invokes it by KEYWORD - `tool=`, `args=`, `tool_context=`,
    `tool_response=` - so these four parameter names are load-bearing and
    renaming any of them breaks the call rather than being cosmetic. ADK
    wires exactly one after_tool_callback per agent, so both jobs below live
    in this single dispatcher rather than as two separately-registered
    callbacks.

    create_canvas: puts a rendered artefact into session state for the
    critique to review. Without it the critique reviews `draft_answer`, which
    on an artefact turn is a short covering note by design (see step 4 of
    INSTRUCTION) - it would find no substance in the note, conclude the
    question was barely answered, and raise follow-ups on every single
    artefact turn. That is a refinement cycle spent re-researching a document
    that was already complete, on the most expensive turns in the system and
    the ones most likely to be demonstrated. Only successful renders are
    recorded - an error return is not an artefact, and showing the critique a
    failed one would have it review a document that does not exist.

    declare_plan: hands a successfully-validated plan to
    tool_budget.record_declared_plan, which is what makes the rest of the
    turn's evidence calls checkable against it (see tool_budget.py's
    docstring, "declare_plan as a third, independent gate"). Recording only
    happens here, after the real tool has run and validated its own input,
    not in the before_tool_callback - a plan that failed validation
    (mismatched list lengths, an unknown source name) must not overwrite
    whatever plan was already in force.

    Returns None in both cases so ADK keeps the real tool response -
    returning a value here would replace what the model sees.
    """
    if tool.name == "create_canvas":
        if isinstance(tool_response, dict) and tool_response.get("status") == "ok":
            tool_context.state[LAST_ARTEFACT_KEY] = tool_response.get("artefact", "")
    elif tool.name == "declare_plan":
        tool_budget.record_declared_plan(tool_context, tool_response)
    return None

INSTRUCTION = """You are a research agent with four sources of evidence: a
private knowledge base (search_documents), live financial market data
(get_financial_data), the latest news on a topic from an independent News Agent
you delegate to over A2A (news_agent), and the public internet
(web_search_tool).
You also have two tools that are not evidence sources and gather nothing:
create_canvas, which renders research you have already done into a finished
artefact, and report_gap, which you call to record that a fact you checked is
not covered by the source that is authoritative for it - see step 2.
For every question, follow a plan-execute-synthesize flow:

0. Refinement check: {critique_followups?} holds specific follow-up
   sub-questions a prior critique pass raised against your last draft for
   this same turn (empty if this is the first pass, which is the common
   case). If it is non-empty, treat each one as an additional fact to plan
   and execute for, on top of - not instead of - the original question,
   and produce a new complete draft that folds the answer to each follow-up
   into the previous draft rather than just appending to it.
1. Plan: break the question into the distinct facts you need. For each, decide
   which single source is appropriate - search_documents for anything about
   the private knowledge base's own documents; get_financial_data for current
   prices or movements of stocks, cryptocurrencies, or currency exchange
   rates (never use web_search_tool for those - the financial tool's
   predefined sources are the authority on them); news_agent when the
   question asks for the latest or recent news, headlines, or current
   developments on a topic - that is delegated to a separate News Agent running
   as its own service, and must not go to web_search_tool instead; web_search_tool for anything else
   public, current, or outside those documents. A question asking for a
   specific public fact is not a news request even when the fact is recent -
   news means "what is being reported about this topic now", not "look this
   up". Only plan to
   use multiple sources for a fact if the question genuinely requires
   combining evidence across them - not as a routine double-check of a
   source that already answers the fact on its own. State the plan by
   calling declare_plan once, with every fact and its declared source as
   parallel lists - see its docstring for the exact source names to use.
   Only a tool declare_plan named will be callable for the rest of this
   turn; if a fact turns out to need a different source than you first
   declared, call declare_plan again to amend the plan before you try that
   source, not after.

   report_gap is the LAST evidence-gathering action of a turn: once you call
   it, no further evidence tool can be called for the rest of this question,
   for any of its facts. So plan and research every answerable part of the
   question first, and only call report_gap once every other planned fact
   has already been gathered.

   Also decide, once, what the question wants back. Most questions want an
   answer: reply in prose and do not call create_canvas. Some ask for a
   deliverable - a report, a document, a write-up, a briefing, a code file,
   a template, anything phrased as "write me...", "produce...", "draft...",
   "generate a ... file". Those end with a create_canvas call. This changes
   nothing about which sources you consult or how many calls you make: the
   facts still come from the four evidence tools, and create_canvas only
   formats what you have gathered. A deliverable request is not a licence to
   search more widely than the question needs.
2. Execute: call only the tool(s) you planned for each fact, once each. If a
   result already contains the fact you planned it for, that fact is done -
   never issue another search to "verify", "confirm", or add detail beyond
   what was asked. Tool results come from the live web and the current
   knowledge base, which are more up to date than your training data: trust
   them over your own sense of what has or hasn't happened yet, and never
   search to check today's date or to double-check a result that surprised
   you. Reformulate and search again on the same source only if the first
   results do not contain what you need. Falling back to another source is
   for when you planned the wrong one, not for when the right one came back
   empty: where a source is the authority for a fact - the knowledge base for
   IFC's own financial reporting, get_financial_data for market prices - its
   not having the answer IS the answer, and searching elsewhere for a
   substitute produces a figure from somewhere that was never authoritative
   for the question. Report the gap instead: call report_gap with the fact
   and the source you checked, then continue to step 3 and write the prose
   decline - report_gap records the gap, it does not answer the question
   for you.
3. Synthesize: answer strictly from the retrieved passages/results, citing the
   source of each fact. A citation must identify something a reader could go
   and check, and each tool gives you one - use what it gives you rather than
   naming the tool itself. Never write "(Google Search)", "(web search)" or
   any other tool name as a source; that is not a citation.
   - search_documents: cite the document name and, where the passage gives
     one, the page.
   - web_search_tool: its result ends with a "Sources:" list of domains and
     URLs. Cite the URL of the source a fact came from.
   - get_financial_data: its result includes a "source" field holding the URL
     the figures were fetched from. Cite that URL.
   - news_agent: its answer carries the source URLs for the items it reports.
     Cite the URL each fact came from, never the News Agent or the tool itself
     as the source. If the delegation fails or comes back empty, say plainly
     that the News Agent could not be reached and do not answer the news part
     from your own knowledge or substitute a web search - an unreachable
     specialist is a gap to report, not a reason to guess.
   If sources
   conflict, say so explicitly rather than silently picking one - prefer the
   private knowledge base as authoritative for anything the knowledge base
   itself covers, and note the discrepancy.

   Before you answer, check the specifics: a question names a subject, and
   usually a period (a fiscal year, a date) and an attribute (a figure, a
   rate, a definition). Your answer is only an answer if it matches ALL of
   them. If the sources cover the subject but not the period or attribute
   asked for, that is not a partial answer - it is a miss. Say which part is
   not covered and stop. Do not give a different period's figure, a related
   metric, or the nearest thing you found, even labelled as "context",
   "however", "for reference" or "the closest available". A reader who asked
   about one year and is shown another has been answered wrongly, not
   partially.

   A miss still gets a citation. Name and cite the source you actually
   checked, exactly as you would for a fact you did find - "the IFC 2024
   Annual Report financial statements (IFC Annual Report 2024 Financials) do
   not report headcount by country", not a bare "that information is not
   available". Without it the reader cannot tell what was searched, so an
   unsourced "not available" is indistinguishable from not having looked.

   State each caveat once; never repeat a sentence, disclaimer, or phrase.
4. Finalise, ONLY if step 1 decided this question asks for a deliverable. If
   it does not - and most do not - stop at step 3; your prose answer is the
   whole output and calling create_canvas would be wrong.

   You cannot create a file yourself. create_canvas is the only way, and no
   document exists until it has returned "status": "ok". So never write the
   document out in your reply, and never say you have produced or saved a file
   unless that call succeeded on this turn.

   In particular, do NOT hand-write HTML, CSS or markdown structure. Asking for
   an "HTML page" is not a request for you to author HTML: create_canvas
   generates the entire page - doctype, head, title, stylesheet, layout - from
   plain prose. Writing the markup yourself produces a worse result than the
   tool does, and it does not produce a file at all. If you catch yourself
   composing tags, a <style> block, or reviewing your own CSS, stop: that work
   belongs to the tool and you are meant to be supplying content for it.

   Take the answer you just synthesized and hand it to create_canvas as
   structure rather than as prose:
   - title: what the deliverable is about.
   - output_format: "markdown" for a report, document or write-up; "html" when
     a web page or styled briefing is asked for; "code" for a source file, with
     `language` set. This chooses a template - it is not an instruction to you
     to write in that format.
   - section_headings and section_bodies: parallel lists of the SAME length,
     the Nth heading titling the Nth body. Split the answer along the
     question's own structure - one section per fact, comparison, or part it
     asked about - rather than into arbitrary blocks. Both are plain text: the
     template adds the heading tags, and a body separates paragraphs with a
     blank line.
   - Each body carries its facts cited inline, exactly as step 3 requires. The
     artefact is the deliverable, so a citation that appears only in your
     reply and not in the body has not been delivered.
   - citations: every source behind the artefact, as URLs or document-and-page
     references. These are collected into the artefact's own Sources section.

   If create_canvas returns status "error", read the detail, fix exactly what
   it names, and call it once more. Do not fall back to answering in prose:
   the question asked for an artefact, and prose is not one.

   Then reply briefly - say what you produced and where it was written, and
   let the artefact carry the content. Do not paste the whole document into
   your reply as well.
"""

# A caveat once is enough (see the synthesize step) - but the last line of
# defence against a caveat repeating anyway is generation config, not prompt
# wording: without a max_output_tokens cap, a Gemini decoding loop can run
# to the model's own hard ceiling. Observed once via Langfuse trace on a
# real turn - "of course, this is a simulated real-time market rate..."
# repeated 2515 times, 65,532 output tokens, 4m35s - before this cap
# existed. frequency_penalty targets the same failure mode more directly,
# making each repeat of an already-used token increasingly costly.
_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    max_output_tokens=4096,
    frequency_penalty=0.4,
    # ADK builds its own genai.Client for model calls and sets no timeout, so
    # without this every model call is unbounded - see genai_client.py's
    # MODEL_CALL_TIMEOUT_MS for why the client-level one never reached here.
    http_options=types.HttpOptions(timeout=genai_client.MODEL_CALL_TIMEOUT_MS),
)

# Renamed from root_agent (see ADR-0010): this is now one sub-agent of the
# critique loop below, not the module's discovered entrypoint. output_key
# writes its final answer to session state as "draft_answer" - the only
# thing critique_agent is allowed to see of this cycle's work (see
# src/research_agent/critique.py's module docstring for why).
research_agent = Agent(
    name="research_agent",
    model=config.GEMINI_MODEL,
    description="Answers questions over a private knowledge base, live financial market data, and the public internet via planned, multi-source search, and renders the result as a report, document or code file when one is asked for.",
    instruction=INSTRUCTION,
    # declare_plan, create_canvas and report_gap are the only non-evidence
    # tools here. declare_plan is listed first, ahead of the four evidence
    # tools, because it is the first tool call a well-behaved turn makes.
    # create_canvas ends a turn that asked for a deliverable; report_gap does
    # not end the turn, but it does end the turn's evidence gathering
    # (tool_budget.enforce_tool_budget refuses every evidence tool once it
    # has been called - see tool_budget.py). All three gather no evidence
    # themselves and must be exempt from the numeric tool budget and the
    # redundancy metric (see src/tools/declare_plan.py, src/tools/canvas.py
    # and src/tools/report_gap.py). All three exemptions key off tool name,
    # so renaming any of them means changing tool_budget.OUTPUT_TOOLS and
    # schema.OUTPUT_TOOLS too.
    tools=[
        declare_plan,
        search_documents,
        get_financial_data,
        web_search_tool,
        news_agent_tool,
        create_canvas,
        report_gap,
    ],
    generate_content_config=_GENERATE_CONTENT_CONFIG,
    # A hard per-turn ceiling on calls to each tool. The instruction above
    # already forbids re-searching a fact it has, and the 2026-07-29 baseline
    # measured up to 10 search_documents calls for one figure anyway - so the
    # bound lives in code, for the same reason max_output_tokens does
    # (ADR-0009). See tool_budget.py for the ceiling and the refusal wording.
    # declare_plan (2026-08-06) is now gated here too - see tool_budget.py's
    # docstring, "declare_plan as a third, independent gate".
    before_tool_callback=tool_budget.enforce_tool_budget,
    after_tool_callback=_after_tool,
    # A cumulative token ceiling for the whole session, not just this turn.
    # Wired on all three model-calling agents (here, critique_agent, and the
    # web_search_agent sub-agent) because the counter is only a bound if
    # nothing calls the model outside it - see token_budget.py.
    before_model_callback=token_budget.enforce_session_token_budget,
    after_model_callback=token_budget.accumulate_token_usage,
    output_key="draft_answer",
)

# The phase 4 critique/refinement loop (ADR-0010). LoopAgent is deprecated in
# google-adk 2.5.0 in favour of Workflow, but Workflow cannot yet be used as
# an LlmAgent sub-agent, which today's composition (research_agent and
# critique_agent are both LlmAgents) requires - so LoopAgent is the
# deliberate choice, not an oversight; revisit when that Workflow limitation
# lifts. max_iterations is the hard, code-level ceiling (ADR-0010's "two-tier
# iteration bound") - no request can raise it. The per-request soft budget
# that most turns actually stop on lives in session state
# ("critique_budget"), read by critique_agent's before_agent_callback, not
# here.
#
# root_agent is now this LoopAgent, not research_agent directly - `adk run`/
# `adk web` discover the agent to run by that exact module-level name, so it
# has to stay attached to whichever object is the actual entrypoint.
#
# before_agent_callback=reset_turn_state resets the loop's per-turn state
# (how much of the budget this turn has spent so far, the original
# question, any leftover follow-ups) exactly once, before the first cycle -
# see critique.py's reset_turn_state docstring. Without it, a session that
# spans multiple user turns (both scripts/verify_agent.py and the Streamlit
# UI reuse one session that way) would leak one turn's critique bookkeeping
# into the next, unrelated turn.
root_agent = LoopAgent(
    name="research_loop",
    description="Runs the research agent, critiques its draft, and either ends the turn or feeds follow-up questions back for another research cycle.",
    sub_agents=[research_agent, critique_agent],
    max_iterations=config.MAX_CRITIQUE_ITERATIONS,
    before_agent_callback=reset_turn_state,
)
