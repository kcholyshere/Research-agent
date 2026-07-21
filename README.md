# Research-agent

An autonomous research agent built incrementally on the Agent Development Kit (ADK): phase 1 is a core RAG agent - a Plan-Execute-Synthesize flow over a private knowledge base, with the Document Search Tool built on the retrieval stack reused from the Finrag project (as the practice mandates).

Status: phase 1, verified against the real GCP project (`gd-gcp-internship-ds`, `gemini-3.5-flash`). Corpus: `ifc-annual-report-2024-financials.pdf` - see `agent_docs/TODOS.md` for the live checklist and `agent_docs/phase_1_requirements.md` for the requirements.

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
