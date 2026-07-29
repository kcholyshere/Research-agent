"""Document Search Tool - the phase 1 key design task.

Wraps the Finrag-derived FAISS retrieval stack as a plain function that ADK
auto-wraps into a tool: the docstring becomes the tool description the LLM
plans against, so it states what the knowledge base contains and what comes
back. Keep it in sync with the actual corpus in data/raw/.
"""

import asyncio
from functools import lru_cache

from src import config
from src.retrieval import faiss_store


@lru_cache(maxsize=1)
def _index():
    # Loaded once per process; FAISS search itself is in-memory (per the
    # phase 1 requirement) and fast at this corpus scale.
    return faiss_store.load_index()


def _search_blocking(query: str) -> list[dict]:
    """The real retrieval, synchronous - always called off the event loop.

    Kept as a separate function rather than inlined so the blocking work has
    one obvious home and the async wrapper stays a wrapper. FAISS's own search
    is fast in memory, but `similarity_search` first embeds the query, and
    `GeminiEmbeddings.embed_query` is a synchronous HTTP call to Vertex - so
    this blocks for a network round trip, not for a vector lookup.
    """
    docs = _index().similarity_search(query, k=config.TOP_K)
    return [{"text": doc.page_content, **doc.metadata} for doc in docs]


async def search_documents(query: str) -> list[dict]:
    """Search the private knowledge base for passages relevant to the query.

    The knowledge base holds one corpus: the International Finance Corporation
    (IFC) 2024 Annual Report financial statements - management's discussion and
    analysis, the consolidated financial statements, and their accompanying
    notes, covering IFC's FY2024 income, assets, investment portfolio, capital
    and accounting policies.

    Use this only for questions about IFC's own financial reporting. A question
    merely being about IFC is not enough: this corpus is one financial report,
    not a source on the organisation. It does not cover what IFC is, who owns
    or governs it, which group it belongs to, where it is headquartered, its
    history, leadership, or any organisational fact outside those FY2024
    statements - those are public knowledge and belong to web search. It
    contains nothing about any other organisation either, and nothing about
    current events, market prices, sport or general knowledge.

    For an unrelated query it returns its nearest passages anyway rather than an
    error, so calling it to confirm that the knowledge base does not cover a
    subject tells you nothing - decide from this description instead.

    Args:
        query: A natural-language question or search phrase.

    Returns:
        The most relevant passages, each with its text and source document,
        ordered most-relevant first.
    """
    # Async purely so the blocking embed-then-search work leaves the event
    # loop free. It used to run inline: ADK calls a sync tool directly rather
    # than in an executor, so every knowledge-base lookup stalled the whole
    # loop for its Vertex embedding round trip. That was invisible while the
    # eval ran one turn at a time, and became a real ceiling once the sweep
    # started running the untimed phase concurrently (ADR-0012) - four
    # concurrent turns cannot overlap through a call that holds the loop.
    # Nothing about the tool's contract or its return shape changes.
    return await asyncio.to_thread(_search_blocking, query)
