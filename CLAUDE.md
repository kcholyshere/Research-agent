# Project Context

## What
An advanced autonomous research agent, built incrementally: phase 1 is a core RAG agent (Plan-Execute-Synthesize over a private knowledge base via a Document Search Tool), later phases add autonomous planning, critique, and refinement of its own research process. Grid University practice series - the capstone project for a Data Science internship's GenAI module; follows on from the Finrag project.

## Why
The series' focus is system architecture and design thinking for agentic systems, not RAG engineering - the RAG internals are a solved input (reused from Finrag per the practice's explicit mandate); the new engineering is the agent around them: ADK agent design, tool composition, planning/critique loops. This should shape how work gets approached: favour discussing and justifying architectural trade-offs over quietly optimising implementation details, even where both would work.

## How
- LLM: `gemini-3.5-flash` via Vertex AI, ADC auth (no API keys), GCP project `gd-gcp-internship-ds` (phase 1 requirements specify Gemini 2.0 Flash, but it was retired from this project's Vertex catalogue by the time of verification - see ADR-0004).
- Framework: Agent Development Kit (ADK) - https://google.github.io/adk-docs/ - agent lives in `src/research_agent/agent.py` (`root_agent`), runnable via `adk run` / `adk web`.
- Retrieval: FAISS in-memory (phase 1 requirement), reusing Finrag's stack - `src/embedding/embedder.py`, `src/retrieval/faiss_store.py`, and `src/services/genai_client.py` are copied from Finrag (commit `69fea4a` era, post-audit) and should be treated as proven; `src/ingestion/chunk.py` is a trimmed adaptation.
- Document Search Tool: `src/tools/document_search.py` - a plain function ADK auto-wraps; its docstring is the tool description the LLM plans against.
- Index build: drop corpus files into `data/raw/`, then `python -m src.dataset` (single embedding pass, from Finrag's A16 pattern).
- Optional UI: `src/ui/app.py`, Streamlit, a thin chat client over `root_agent` via `google.adk.runners.InMemoryRunner` (see ADR-0002). Containerisation: Docker (Dockerfile present, unverified until phase 1 has a corpus).

## Critical rules
- Commit and push at reasonable intervals (per logical step, not one batch at the end).
- Requirements per phase live in `agent_docs/phase_N_requirements.md`; keep `agent_docs/TODOS.md` current as phases progress.
- The moment a TODO item is actually fixed or verified, check it off in `agent_docs/TODOS.md` in the same turn - don't batch updates for later. Keep each entry to one to two lines max; detail belongs in commit messages, code comments, or `agent_docs/decisions.md`, not here.
- The RAG stack (`src/embedding/embedder.py`, `src/retrieval/faiss_store.py`, `src/services/genai_client.py`, `src/ingestion/chunk.py`) is reused from the previous module's project (Finrag), not built for this one - see ADR-0000. If you notice leftovers that don't fit this project (Finrag-specific naming, comments, config, or machinery like hybrid retrieval/reranking/ColPali/table-image handling), do not silently strip or change them - flag it and propose what to do with it (keep as harmless unused code, trim, or replace).
- For significant architectural decisions, append an entry to `agent_docs/decisions.md` using the template convention there (context, options considered, decision, consequences, transferable principle).
- `google-adk` is pinned (`==2.5.0`, ADR-0002). The Runner/session API actually in the installed package differs from most docs/blog snippets - `InMemoryRunner(agent=, app_name=)` + `await runner.session_service.create_session(...)`; verify against `inspect.signature()` on the real package, not just docs, if it moves again.
- Agent + UI mechanics were first verified (`InMemoryRunner`/`run_async`, `search_documents` tool call, Streamlit `AppTest`) against a substitute GCP project/model (`gd-gcp-gridu-genai` / `gemini-3.5-flash`), since ADC had no access yet to phase 1's actual project. ADC now has access to `gd-gcp-internship-ds` and a direct `generate_content` call against it succeeds with `gemini-3.5-flash` (2026-07-21) - but the full agent + UI flow (Runner, tool call, Streamlit) has only been re-run against the real project informally, not re-verified with the same rigour as the original substitute-project pass. Do that properly once a corpus exists.
- The phase 1 knowledge-base corpus is not yet chosen - that is the first real decision of the project (log it as ADR-0001).
