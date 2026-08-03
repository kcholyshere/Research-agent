# Research-agent
An autonomous research agent built incrementally on the Agent Development Kit (ADK): phase 1 is a core RAG agent - a Plan-Execute-Synthesize flow over a private knowledge base, with the Document Search Tool built on the retrieval stack reused from the Finrag project (as the practice mandates).

Status: phase 1, verified against the real GCP project (`gd-gcp-internship-ds`, `gemini-3.5-flash`). Corpus: `ifc-annual-report-2024-financials.pdf` - see `agent_docs/TODOS.md` for the live checklist and `agent_docs/phase_1_requirements.md` for the requirements.

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
├── tools/document_search.py   <- the Document Search Tool (ADK function tool)
├── research_agent/agent.py    <- ADK root_agent, Plan-Execute-Synthesize
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

## Phase 5: News Agent (A2A demo)
A minimal, single-purpose News Agent runs as its own process, reached over a
plain HTTP endpoint rather than the `a2a` protocol library named in the phase
1 requirements (not installed in this venv; see `src/news_service/server.py`'s
docstring for why HTTP was chosen instead). Demonstrates delegation across a
process boundary: the main agent's `get_latest_news` tool
(`src/tools/news_agent.py`) never imports the News Agent - it only ever
speaks to it over the network, and degrades to a clear error dict if that
service is not running.

```bash
# terminal 1 - start the News Agent service (binds :8001)
uv run python -m src.news_service.server

# terminal 2 - demo it directly
curl -s -X POST http://localhost:8001/news \
    -H "Content-Type: application/json" \
    -d '{"topic": "artificial intelligence regulation"}' | python -m json.tool

# or drive it from the main agent (once wired into research_agent/agent.py)
uv run adk web src
```
