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
- Not containerised yet; it runs from a local checkout.
- News questions need the News Agent process running separately.
- Knowledge-base answers do not yet carry inline citations.
- Repeats searches on some questions, up to six calls.
- Sometimes answers from the web instead of declining cleanly.
- Web citations are Vertex redirect links, not readable URLs.
- No turn timeout outside the evaluation harness; a turn can hang.
- The critique loop nearly always stops after one cycle.
- Single corpus, rebuilt by hand when documents change.

## Running it locally
One-time setup:
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

Start the News Agent, which serves as its own A2A process on port 8001:
```bash
uv run python -m src.news_service.server
```
Leave it running in its own terminal. Every other entrypoint starts fine without
it, because the agent card is resolved lazily; only news questions fail, and they
fail by reporting that the specialist was unreachable rather than by quietly
falling back to a web search. To see what it advertises:
```bash
curl -s http://localhost:8001/.well-known/agent-card.json | python -m json.tool
```

Then start the agent, in a second terminal, whichever way suits:
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
├── news_service/              <- the News Agent, served as its own A2A process
├── research_agent/agent.py    <- research_agent + root_agent (the critique loop)
├── research_agent/critique.py <- the critique agent and loop control
├── research_agent/tool_budget.py <- per-turn ceiling on evidence calls
├── evaluation/                <- question set, runner, metrics, replay layer
└── ui/app.py                  <- Streamlit chat UI over root_agent
```

Project tracking lives in `agent_docs/`: `TODOS.md` for the live checklist and
`decisions.md` for the architectural decision log.
