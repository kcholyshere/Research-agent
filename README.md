# Research-agent
An autonomous research agent built on the Agent Development Kit (ADK). It plans a
question into sub-questions, gathers evidence from four independent sources,
critiques its own draft, and either answers in prose or finalises the result into
a document you can download.

The knowledge base is IFC's 2024 annual report financials. Retrieval runs on
FAISS, the model is `gemini-3.5-flash` on Vertex AI, and every run is traced to
Langfuse.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the pieces fit together, and
[`EVALUATION.md`](EVALUATION.md) for measured results.

## Functionalities
- Answers questions from the private IFC corpus using FAISS retrieval.
- Searches the public web when the corpus cannot answer.
- Fetches live stock, crypto and currency prices over MCP.
- Delegates news questions to an independent News Agent over A2A.
- Plans multi-source questions and combines evidence from several tools.
- Critiques its own draft each turn and refines within a budget.
- Produces markdown reports, standalone HTML pages, or commented code files.
- Cites its sources, and says plainly when a fact is absent.
- Traces every run to Langfuse for diagnosis.
- Runs from a chat UI, ADK's dev UI, or the terminal.

## Limitations
- News and financial questions need their services running; the rest degrades cleanly.
- Sometimes delegates to the News Agent when only the web is needed.
- Repeats searches on some questions, up to six calls.
- Sometimes answers from the web instead of declining cleanly.
- Web citations are Vertex redirect links, not readable URLs.
- No turn timeout outside the evaluation harness; a turn can hang.
- A session stops answering at 200,000 cumulative tokens (`MAX_SESSION_TOKENS`); history is not trimmed, so each turn costs more than the last.
- The critique loop nearly always stops after one cycle.
- Single corpus, rebuilt by hand when documents change.

## Setup
One-time, and needed either way you run it:
```bash
uv sync
cp .env.example .env   # project: gd-gcp-internship-ds
gcloud auth application-default login
```

Build the index (once, and again whenever `data/raw/` changes):
```bash
# drop corpus files (*.txt, *.md, *.pdf) into data/raw/, then:
uv run python -m src.dataset
```
This is a host-side step in both cases. The image ships no corpus and no index -
compose bind-mounts `data/` and `models/` instead, so a container started before
this has run will answer knowledge-base questions from an empty index.

## Running it with Docker
The whole system is four services (see ADR-0020):
```bash
docker compose up --build
```
- <http://localhost:8501> - the Streamlit chat UI
- <http://localhost:8000> - ADK's dev UI
- `news-agent` on 8001 and `mcp-fetch` on 8090, which the agent reaches by
  service name and you can reach on those ports for debugging

Your `~/.config/gcloud` is mounted read-only for Application Default
Credentials, so no key material goes into the image or into `docker-compose.yml`.

## Running it locally
Start the two services the agent depends on, then the agent itself. Both are
containers, and you can start them without the rest of the stack:
```bash
docker compose up -d mcp-fetch news-agent
```
`mcp-fetch` is Anthropic's reference MCP fetch server fronted by a stdio-to-HTTP
proxy - the financial route needs it, and without it financial questions report
the server as unreachable rather than answering from another source.

`news-agent` serves the A2A agent card on 8001. Every entrypoint starts fine
without it, because the card is resolved lazily; only news questions fail, and
they fail by reporting the specialist unreachable rather than quietly falling
back to a web search. To see what it advertises:
```bash
curl -s http://localhost:8001/.well-known/agent-card.json | python -m json.tool
```
The RPC address in that card is derived from the `Host` header you reached it
on, so the same service tells this shell `localhost:8001` and tells the agent
container `news-agent:8001` - a local checkout and a container get an address
that works for each. See ADR-0022 for why, and for where that stops being safe.

Then start the agent, whichever way suits:
```bash
uv run python -m streamlit run src/ui/app.py   # chat UI, recommended
uv run adk web src                             # ADK's own dev UI
uv run adk run src/research_agent              # terminal
```

Use `python -m streamlit`, not the `streamlit` console-script shim - the shim's
shebang hard-codes the venv's absolute path at `uv sync` time, so it breaks if the
project directory is renamed or moved without recreating `.venv`. `python -m`
resolves through the interpreter instead.

Things worth asking, one per capability:
```text
What was IFC's Net Income for the fiscal year ending June 30, 2024?
Who is the current President of the World Bank Group?
What is the current price of Bitcoin?
What is the latest news about the International Finance Corporation?
Write me a short markdown report on IFC's FY24 net income and total assets.
```

## Evaluation
The harness runs from the local checkout, and since ADR-0020 the financial
questions go through the `mcp-fetch` service - so start it first or every
financial question fails:
```bash
docker compose up -d mcp-fetch news-agent
```
```bash
# full sweep (about an hour: 35 questions x 2 arms x 4 reps)
uv run python -m src.evaluation.run_eval --reps 4 --mode live --concurrency 4

# a fast smoke over named questions
uv run python -m src.evaluation.run_eval \
    --questions kb-net-income,canvas-kb-report --reps 1 --mode live --budgets 0
```
The harness scores routing, redundancy, citation, decline, content, artefact and
wasted-cycle assertions deterministically, with no LLM judge - see ADR-0011 for
why. Results and their interpretation live in `EVALUATION.md`.

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
├── tools/document_search.py   <- Document Search Tool, FAISS
├── tools/web_search.py        <- Web Search Tool, google_search sub-agent
├── tools/financial_data.py    <- Financial Data Tool over MCP
├── tools/news_agent.py        <- A2A client for the News Agent
├── tools/canvas.py            <- Canvas: renders research into an artefact
├── tools/declare_plan.py      <- one authoritative source per fact, gated against
├── tools/report_gap.py        <- explicit stop: source checked, fact absent
├── news_service/              <- the News Agent, served as its own A2A process
├── research_agent/agent.py    <- research_agent + root_agent (the critique loop)
├── research_agent/critique.py <- the critique agent and loop control
├── research_agent/tool_budget.py <- the three tool gates: ceiling, plan, gap
├── research_agent/token_budget.py <- cumulative token ceiling for a session
├── research_agent/turn_deadline.py <- wall-clock bound on a turn
├── research_agent/history_trim.py <- keeps a turn from re-sending the whole chat
├── evaluation/                <- question set, runner, metrics, replay layer
└── ui/app.py                  <- Streamlit chat UI over root_agent
```

Project tracking lives in `agent_docs/`: `TODOS.md` for the live checklist and
`decisions.md` for the architectural decision log.
