# ResearchAgent

An autonomous research agent built incrementally on the Agent Development Kit (ADK): phase 1 is a core RAG agent - a Plan-Execute-Synthesize flow over a private knowledge base, with the Document Search Tool built on the retrieval stack reused from the Finrag project (as the practice mandates).

Status: phase 1 scaffold. The agent skeleton is written against the ADK docs but unverified until dependencies are installed and a knowledge-base corpus is chosen - see `agent_docs/TODOS.md` for the live checklist and `agent_docs/phase1-requirements.md` for the requirements.

## Layout

```
src/
├── config.py                  <- GCP project/model IDs, paths, chunking, top-k
├── dataset.py                 <- index build: data/raw/ -> chunks -> FAISS
├── ingestion/chunk.py         <- chunking + JSONL persistence (trimmed from Finrag)
├── embedding/embedder.py      <- GeminiEmbeddings via Vertex AI (from Finrag)
├── retrieval/faiss_store.py   <- FAISS HNSW build/load (from Finrag)
├── services/genai_client.py   <- shared Vertex AI client (from Finrag)
├── tools/document_search.py   <- the Document Search Tool (ADK function tool)
├── research_agent/agent.py    <- ADK root_agent, Plan-Execute-Synthesize
└── ui/app.py                  <- optional Streamlit chat UI over root_agent
```

## Setup (agent + UI mechanics verified against a substitute GCP project - see `agent_docs/TODOS.md`)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # project: gd-gcp-internship-ds
gcloud auth application-default login
# drop corpus files (*.txt, *.md) into data/raw/, then:
python -m src.dataset
adk web src             # ADK's own dev UI, or: adk run src/research_agent
streamlit run src/ui/app.py   # the optional chat UI
```
