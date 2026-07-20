# Project Context

## What
An advanced autonomous research agent, built incrementally: phase 1 is a core RAG agent (Plan-Execute-Synthesize over a private knowledge base via a Document Search Tool), later phases add autonomous planning, critique, and refinement of its own research process. Grid University practice series; follows on from the Finrag project.

## Why
The series' focus is system architecture and design thinking for agentic systems - the RAG internals are a solved input (reused from Finrag per the practice's explicit mandate); the new engineering is the agent around them: ADK agent design, tool composition, planning/critique loops.

## How
- LLM: Gemini 2.0 Flash via Vertex AI, ADC auth (no API keys), GCP project `gd-gcp-internship-ds`.
- Framework: Agent Development Kit (ADK) - https://google.github.io/adk-docs/ - agent lives in `src/research_agent/agent.py` (`root_agent`), runnable via `adk run` / `adk web`.
- Retrieval: FAISS in-memory (phase 1 requirement), reusing Finrag's stack - `src/embedding/embedder.py`, `src/retrieval/faiss_store.py`, and `src/services/genai_client.py` are copied from Finrag (commit `69fea4a` era, post-audit) and should be treated as proven; `src/ingestion/chunk.py` is a trimmed adaptation.
- Document Search Tool: `src/tools/document_search.py` - a plain function ADK auto-wraps; its docstring is the tool description the LLM plans against.
- Index build: drop corpus files into `data/raw/`, then `python -m src.dataset` (single embedding pass, from Finrag's A16 pattern).
- Optional UI: `src/ui/app.py`, Streamlit, a thin chat client over `root_agent` via `google.adk.runners.InMemoryRunner` (see ADR-0002). Containerisation: Docker (Dockerfile present, unverified until phase 1 has a corpus).

## Critical rules
- Commit and push at reasonable intervals (per logical step, not one batch at the end).
- Requirements per phase live in `agent_docs/phaseN-requirements.md`; keep `agent_docs/TODOS.md` current as phases progress.
- For significant architectural decisions, append an entry to `agent_docs/decisions.md` using the template convention there (context, options considered, decision, consequences, transferable principle).
- `google-adk` is pinned (`==2.5.0`, ADR-0002). The Runner/session API actually in the installed package differs from most docs/blog snippets - `InMemoryRunner(agent=, app_name=)` + `await runner.session_service.create_session(...)`; verify against `inspect.signature()` on the real package, not just docs, if it moves again.
- Agent + UI mechanics are verified (`InMemoryRunner`/`run_async`, `search_documents` tool call, Streamlit `AppTest`) but only against a substitute GCP project/model (`gd-gcp-gridu-genai` / `gemini-3.5-flash`) - this machine's ADC has no access yet to phase 1's actual project (`gd-gcp-internship-ds`) or model (`gemini-2.0-flash`). Re-verify once that access exists.
- The phase 1 knowledge-base corpus is not yet chosen - that is the first real decision of the project (log it as ADR-0001).
