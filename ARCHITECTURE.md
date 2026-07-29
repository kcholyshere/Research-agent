# Architecture
High-level map of the system: one component diagram (what talks to what) and one
sequence diagram (what happens during a single question). Both stay at the same
level of granularity - the shape of the system, not its internals. For file-by-file
detail see the [Layout section of the README](README.md#layout); for why things are
built this way, see [`agent_docs/decisions.md`](agent_docs/decisions.md).

## Components
```mermaid
flowchart TD
    subgraph Entry["Entrypoints"]
        CLI["adk run / adk web<br/>(ADK CLI)"]
        UI["Streamlit UI"]
    end

    Agent["root_agent<br/>Plan → Execute → Synthesize<br/>(single Gemini agent, one instruction)"]

    subgraph Tools["Tools"]
        DocTool["Document Search Tool"]
        FinTool["Financial Data Tool"]
        WebTool["Web Search Tool<br/>(sub-agent + google_search)"]
    end

    subgraph KB["Knowledge base (built offline via python -m src.dataset)"]
        Raw["data/raw/*.pdf"]
        Parse["Parse (Docling)"]
        Chunk["Chunk"]
        Embed["Embed (Gemini embeddings)"]
        Index["FAISS index (in-memory)"]
    end

    subgraph Ext["External services"]
        Vertex["Vertex AI<br/>(chat + embedding models)"]
        MCP["MCP fetch server<br/>(Docker, spawned per call)"]
        Yahoo["Yahoo Finance<br/>(3 hardcoded pages)"]
        GSearch["Google Search grounding"]
    end

    Obs["Langfuse tracing<br/>(instruments every entrypoint)"]

    CLI --> Agent
    UI --> Agent
    Agent -. instrumented by .-> Obs

    Agent --> DocTool --> Index
    Agent --> FinTool --> MCP --> Yahoo
    Agent --> WebTool --> GSearch
    Agent --> Vertex

    Raw --> Parse --> Chunk --> Embed --> Index
    Embed --> Vertex

    click CLI "README.md#setup" "adk CLI usage"
    click UI "src/ui/app.py" "Streamlit chat UI"
    click Agent "src/research_agent/agent.py" "root_agent: instruction + tool wiring"
    click DocTool "src/tools/document_search.py" "Document Search Tool"
    click FinTool "src/tools/financial_data.py" "Financial Data Tool"
    click WebTool "src/tools/web_search.py" "Web Search Tool"
    click Raw "data/raw" "corpus files"
    click Parse "src/ingestion/parse.py" "PDF parsing (Docling)"
    click Chunk "src/ingestion/chunk.py" "chunking + JSONL persistence"
    click Embed "src/embedding/embedder.py" "GeminiEmbeddings"
    click Index "src/retrieval/faiss_store.py" "FAISS build/load"
    click Vertex "src/services/genai_client.py" "shared Vertex AI client"
```

| Component | File |
|---|---|
| Entrypoints (CLI) | `adk run` / `adk web` - see [README Setup](README.md#setup) |
| Entrypoints (UI) | [`src/ui/app.py`](src/ui/app.py) |
| root_agent (plan-execute-synthesize) | [`src/research_agent/agent.py`](src/research_agent/agent.py) |
| Document Search Tool | [`src/tools/document_search.py`](src/tools/document_search.py) |
| Financial Data Tool | [`src/tools/financial_data.py`](src/tools/financial_data.py) |
| Web Search Tool | [`src/tools/web_search.py`](src/tools/web_search.py) |
| Corpus | [`data/raw/`](data/raw) |
| Parse (Docling) | [`src/ingestion/parse.py`](src/ingestion/parse.py) |
| Chunk | [`src/ingestion/chunk.py`](src/ingestion/chunk.py) |
| Embed | [`src/embedding/embedder.py`](src/embedding/embedder.py) |
| FAISS index | [`src/retrieval/faiss_store.py`](src/retrieval/faiss_store.py) |
| Index build entrypoint | [`src/dataset.py`](src/dataset.py) |
| Shared Vertex AI client | [`src/services/genai_client.py`](src/services/genai_client.py) |
| Config (models, paths, GCP project) | [`src/config.py`](src/config.py) |

GitHub strips `click` bindings from rendered Mermaid diagrams (security sandboxing), so the
table above is the reliable link path there; `click` works in editors/tools that render
Mermaid with default settings (e.g. the Mermaid Live Editor, most IDE previews).

## One turn, end to end
```mermaid
sequenceDiagram
    participant User
    participant Agent as root_agent
    participant Tool as Document/Financial/Web Tool
    participant Langfuse

    User->>Agent: question
    activate Agent
    Agent->>Langfuse: trace start (span tree)
    loop per fact needed (plan → execute)
        Agent->>Agent: plan: which single tool for this fact?
        Agent->>Tool: call
        Tool-->>Agent: result
    end
    Agent->>Agent: synthesize: answer strictly from results, cite sources
    Agent-->>User: final answer
    deactivate Agent
```

The three tools are mutually exclusive per fact by design (see the planner instruction in
`agent.py`): knowledge-base questions go to Document Search, live stock/crypto/currency
questions go to the Financial Data Tool, everything else public goes to Web Search. A fact
only uses more than one tool when the question genuinely requires combining evidence across
sources.
