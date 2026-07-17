"""Document Search Tool - the phase 1 key design task.

Wraps the Finrag-derived FAISS retrieval stack as a plain function that ADK
auto-wraps into a tool: the docstring becomes the tool description the LLM
plans against, so it states what the knowledge base contains and what comes
back. Keep it in sync with the actual corpus in data/raw/.
"""

from functools import lru_cache

from src import config
from src.retrieval import faiss_store


@lru_cache(maxsize=1)
def _index():
    # Loaded once per process; FAISS search itself is in-memory (per the
    # phase 1 requirement) and fast at this corpus scale.
    return faiss_store.load_index()


def search_documents(query: str) -> list[dict]:
    """Search the private knowledge base for passages relevant to the query.

    Args:
        query: A natural-language question or search phrase.

    Returns:
        The most relevant passages, each with its text and source document,
        ordered most-relevant first.
    """
    docs = _index().similarity_search(query, k=config.TOP_K)
    return [{"text": doc.page_content, **doc.metadata} for doc in docs]
