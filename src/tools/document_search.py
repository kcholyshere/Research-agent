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


def _index_files_present() -> bool:
    """Whether both files `load_index` reads are actually on disk.

    Checked directly rather than by inspecting what `FAISS.load_local` raises
    when they are not, because that exception is not a clean discriminator.
    Verified against the installed `faiss` package: a missing `.faiss` file
    surfaces as a `RuntimeError` from FAISS's own C++ layer ("could not open
    ... for reading: No such file or directory"), and a *corrupt* `.faiss`
    file - present, but not a valid index - raises the exact same
    `RuntimeError` type with a different message ("Index type ... not
    recognized"). A missing `.pkl` (the docstore half of the pair) raises a
    plain `FileNotFoundError` instead, a third shape again. Catching
    `RuntimeError` and pattern-matching its text to tell "absent" from
    "corrupt" apart would be exactly the brittle string-matching trade-off
    `news_agent.py` explicitly declined for its own timeout case (see that
    module's docstring); checking existence up front is precise where
    message-sniffing is not, and it also catches the partial-write case (one
    file present, the other missing) without a second exception type to
    handle.
    """
    return all(
        (config.FAISS_INDEX_DIR / f"{faiss_store.INDEX_NAME}{suffix}").exists()
        for suffix in (".faiss", ".pkl")
    )


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
        ordered most-relevant first. If retrieval fails, instead returns a
        single-item list holding an "error" key describing what went wrong
        and what to do next - it does not raise. This corpus is the only
        source for IFC's own FY2024 reporting, so if the error persists on
        retry, call report_gap for the fact rather than answering the
        question from another source; a web search cannot substitute for it
        honestly.
    """
    # Async purely so the blocking embed-then-search work leaves the event
    # loop free. It used to run inline: ADK calls a sync tool directly rather
    # than in an executor, so every knowledge-base lookup stalled the whole
    # loop for its Vertex embedding round trip. That was invisible while the
    # eval ran one turn at a time, and became a real ceiling once the sweep
    # started running the untimed phase concurrently (ADR-0012) - four
    # concurrent turns cannot overlap through a call that holds the loop.
    # Nothing about the tool's contract or its normal-path return shape
    # changes; only the failure path below is new.
    if not _index_files_present():
        # The exact failure the audit measured live: `models/` stopped being
        # tracked at commit 1d91762 and nothing rebuilds it automatically, so
        # every knowledge-base question died with a raw FAISS.load_local
        # traceback out of runner.run_async (no on_tool_error_callback is
        # registered anywhere in this project, verified by grep, so nothing
        # catches it upstream). This is a setup problem, not a transient one -
        # telling the model "try again" here would be a false diagnosis, so
        # the message instead names the one command that fixes it and does
        # not invite a retry.
        #
        # It points the model at report_gap by name rather than at web
        # search, even though report_gap's own docstring frames it as "you
        # consulted the source and it does not contain the fact" - not
        # "the source is unbuildable right now", which is what actually
        # happened. That is a real mismatch, accepted deliberately: there is
        # no tool in this project for "the authoritative source could not be
        # reached at all", and of the two available reactions - substitute an
        # unauthorised source, or stop and say so - stopping is the one
        # step 2 of the instruction already asks for, and report_gap is the
        # only first-class action that stops rather than merely asks nicely
        # (see tool_budget.py's terminal-gate comment). Reporting a gap for
        # the wrong reason is a smaller error than answering from a source
        # that was never authoritative for this fact.
        return [
            {
                "error": (
                    "The knowledge base index has not been built - no FAISS "
                    f"files exist under {config.FAISS_INDEX_DIR}. This is a "
                    "setup problem that will not clear on its own, so "
                    "retrying will not help. Build it with `python -m "
                    "src.dataset`, then retry this query. Nothing else in "
                    "this system covers the IFC FY2024 report, so if this "
                    "fact is needed now, call report_gap for it rather than "
                    "answering from another source."
                )
            }
        ]

    try:
        return await asyncio.to_thread(_search_blocking, query)
    except Exception as exc:  # noqa: BLE001 - degrade rather than kill the turn, see below
        # Everything that reaches here is NOT "the index files are absent"
        # (that was ruled out above) - it is a corrupt or truncated .faiss/
        # .pkl, a dimension mismatch between the stored index and
        # GeminiEmbeddings, or - the case the audit calls out by name - a
        # transient Vertex 429/503 raised synchronously inside
        # GeminiEmbeddings.embed_query while embedding the query. Unlike the
        # missing-index case, this class of failure may well clear on its
        # own, so the wording below allows a retry and deliberately does NOT
        # point at `python -m src.dataset` - doing so would misdiagnose a
        # live service blip as a missing build step, which is the opposite
        # mistake to the one this fix exists to prevent. It still names
        # report_gap by name rather than a substitute tool, for the same
        # reason and with the same accepted mismatch as the missing-index
        # case above: this corpus is the only source for IFC's own FY2024
        # reporting, so a confident answer from web search after this fails
        # would be answering a different question and presenting it as this
        # one, and stopping via report_gap is the smaller error of the two.
        return [
            {
                "error": (
                    f"Knowledge base search failed ({type(exc).__name__}: "
                    f"{exc}). The index itself is present, so this is likely "
                    "a passing failure (for example a Vertex API error) or a "
                    "corrupted index, and retrying once may succeed. If it "
                    "keeps failing, call report_gap for the fact rather than "
                    "answering from another source: no other tool covers "
                    "the IFC FY2024 report."
                )
            }
        ]
