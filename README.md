# Research-agent
An autonomous research agent built incrementally on the Agent Development Kit (ADK): phase 1 is a core RAG agent - a Plan-Execute-Synthesize flow over a private knowledge base, with the Document Search Tool built on the retrieval stack reused from the Finrag project (as the practice mandates).

Status: phases 1-6 implemented, verified against the real GCP project (`gd-gcp-internship-ds`, `gemini-3.5-flash`). Corpus: `ifc-annual-report-2024-financials.pdf`. The agent plans over four evidence sources - the private knowledge base, live financial data over MCP, the public web, and an independent News Agent reached over A2A - critiques its own draft in a loop, and can finalise the result into a report, document or code file. See `agent_docs/TODOS.md` for the live checklist, `agent_docs/decisions.md` for the architectural decisions, and `EVALUATION.md` for measured results.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a diagram of how the pieces fit together.

## Layout
```
src/
├── config.py                  <- GCP project/model IDs, paths, chunking, top-k
├── dataset.py                 <- index build: data/raw/ -> chunks -> FAISS
├── ingestion/parse.py         <- PDF parsing via Docling, trimmed from Finrag
├── ingestion/chunk.py         <- chunking + JSONL persistence (trimmed from Finrag)
├── embedding/embedder.py      <- GeminiEmbeddings via Vertex AI (from Finrag)
├── retrieval/faiss_store.py   <- FAISS HNSW build/load (from Finrag)
├── services/genai_client.py   <- shared Vertex AI client (from Finrag)
├── tools/document_search.py   <- Document Search Tool, FAISS (phase 1)
├── tools/web_search.py        <- Web Search Tool, google_search sub-agent (phase 2)
├── tools/financial_data.py    <- Financial Data Tool over MCP (phase 3)
├── tools/news_agent.py        <- A2A client for the News Agent (phase 5)
├── tools/canvas.py            <- Canvas: renders research into an artefact (phase 6)
├── news_service/             <- the News Agent, served as its own A2A process
├── research_agent/agent.py    <- research_agent + root_agent (the critique loop)
├── research_agent/critique.py <- the critique agent and loop control (phase 4)
├── research_agent/tool_budget.py <- per-turn ceiling on evidence calls
├── evaluation/               <- question set, runner, metrics, replay layer
└── ui/app.py                  <- optional Streamlit chat UI over root_agent
```

## Setup
```bash
uv sync
cp .env.example .env   # project: gd-gcp-internship-ds
gcloud auth application-default login
# drop corpus files (*.txt, *.md, *.pdf) into data/raw/, then:
uv run python -m src.dataset
uv run adk web src             # ADK's own dev UI, or: uv run adk run src/research_agent
uv run python -m streamlit run src/ui/app.py   # the optional chat UI
```

Use `python -m streamlit`, not the `streamlit` console-script shim - the shim's shebang
hard-codes the venv's absolute path at `uv sync` time, so it breaks if the project
directory is ever renamed/moved without recreating `.venv` (hit this after the
`ResearchAgent` -> `Research-agent` rename). `python -m` resolves through the
interpreter instead, so it's rename-proof.

## Phase 5: News Agent (A2A delegation)
A minimal, single-purpose News Agent runs as **its own process**, reached over the
**A2A protocol** rather than a bespoke HTTP endpoint. The main agent talks to it
through `news_agent` (`src/tools/news_agent.py`), a `RemoteA2aAgent` wrapped in an
`AgentTool`: it never imports the News Agent, only ever addresses it across a
process boundary.

Both halves come from ADK itself - `to_a2a` serves any ADK agent as an A2A app,
and `RemoteA2aAgent` consumes one - so this needs the `[a2a]` extra
(`google-adk[a2a]`, already in `pyproject.toml`) and no bespoke transport code.
See ADR-0018 in `agent_docs/decisions.md`; ADR-0017 records an earlier plain-HTTP
version and why it was replaced.

The agent is **discovered rather than hard-coded**: the client is pointed at the
agent card, and learns the RPC URL and the agent's skills from it.

```bash
# terminal 1 - start the News Agent as its own A2A service (binds :8001)
uv run python -m src.news_service.server

# terminal 2 - inspect the agent card it publishes
curl -s http://localhost:8001/.well-known/agent-card.json | python -m json.tool

# then ask the main agent a news question, e.g.
#   "What is the latest news about the International Finance Corporation?"
uv run python -m streamlit run src/ui/app.py    # or: uv run adk web src
```

**Demo prerequisite:** the News Agent must be running, or news questions fail.
That is by design rather than a rough edge - the agent reports that the
specialist could not be reached instead of quietly falling back to a web search,
because an unreachable source is a gap to report, not a reason to guess. The
agent card is resolved lazily on first use, so every other entrypoint still
starts normally when the service is down; only news questions are affected.

## Phase 6: Canvas
`create_canvas` (`src/tools/canvas.py`) turns research the agent has already
gathered into a finished artefact. It is the only tool here that *produces*
rather than retrieves, which is why it is exempt from the per-turn tool ceiling
and from the redundancy metric - see ADR-0016.

Three formats: **markdown** (a report or document), **html** (a complete styled
standalone page), and **code** (a commented source file, with the language
driving both the comment token and the file extension). The tool generates all
the markup itself from plain prose - Pydantic validates the request structure,
Jinja2 renders it.

Artefacts are written to `data/processed/artefacts/` and also returned inline, so
they render in the Streamlit UI with a download button. Ask for one with a
deliverable-shaped request:

> Write me a short markdown report on IFC's FY24 net income and total assets,
> with one section for each.

A plain question is still answered in prose - the agent decides once, during
planning, whether the question wants an answer or a deliverable.

## Evaluation
```bash
# full sweep (~1 hour for 35 questions x 2 arms x 4 reps)
PYTHONPATH=. uv run python -m src.evaluation.run_eval --reps 4 --mode live --concurrency 4

# a fast smoke over named questions
PYTHONPATH=. uv run python -m src.evaluation.run_eval \
    --questions kb-net-income,canvas-kb-report --reps 1 --mode live --budgets 0
```
Results and their interpretation live in `EVALUATION.md`. The harness scores
routing, redundancy, citation, decline, content, artefact and wasted-cycle
assertions deterministically, with no LLM judge - see ADR-0011 for why.
