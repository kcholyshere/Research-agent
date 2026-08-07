"""The error path in `src/tools/document_search.py` (`agent_docs/audit.md`, finding 2).

## What this pins

Before this fix, `search_documents` had no error handling at all: a missing
FAISS index (the live state of this checkout - `models/` stopped being
tracked at commit `1d91762` and there is no `.faiss` file anywhere in the
repository), a corrupted-but-present index, or a transient failure inside
`GeminiEmbeddings.embed_query` (a Vertex 429/503) all propagated straight out
of `runner.run_async` as a raw traceback, because no `on_tool_error_callback`
is registered anywhere in this project (verified by grep, same as the
module's own comment records). `financial_data.py` and `news_agent.py` both
degrade instead of raising for their own unreachable-service cases; this
brings `search_documents` in line.

The three failure shapes need genuinely different wording, not merely
non-empty wording, mirroring the same principle `test_news_agent_recovery.py`
pins for its two shapes: each test below asserts ITS OWN diagnostic phrase is
present *and* the other shape's phrase is absent, rather than banning a
single word - a word-ban (e.g. "assert 'transient' not in message") would
pass on a message that says something equally wrong in different words, e.g.
"just try again in a minute". The two positive phrases
(`RETRY_WONT_HELP_PHRASE`, `RETRY_MAY_HELP_PHRASE`) are mutually exclusive by
construction, so each test's assertion is checking an actual claim rather
than the absence of one string.

## The three shapes, and why the third (corrupt index) needs its own test

`_index_files_present`'s whole justification (see its docstring in
`document_search.py`) is that a *missing* `.faiss` file and a *corrupt* one
raise the exact same `RuntimeError` type from FAISS's C++ layer, only
distinguished by message text - which is why the fix checks file existence
directly rather than pattern-matching that text. That claim is only real if
something exercises the corrupt case: `test_corrupt_index_is_not_mistaken_for_a_missing_one`
writes non-index bytes into both files and confirms the result still lands
in the "index is present, this may clear on retry" bucket, not the "go run
`python -m src.dataset`" bucket - i.e. that a corrupt-but-present index is
not misdiagnosed as an absent one.

## Why a real tiny FAISS index rather than mocking `faiss_store` wholesale

The "other failure" tests need the code to get *past* the missing-index
check and fail later, inside the actual load or search. Building one real
chunk through `faiss_store.build_index`/`save_index` and then only patching
`GeminiEmbeddings.embed_query` (the exact call the audit names, for the
transient-failure test) exercises the real `FAISS.load_local` /
`similarity_search` machinery for everything except the one network call
being simulated, rather than asserting against a mock of the code under test.

## Confirming this is a real regression test, not a vacuous one

All three tests were run against the pre-fix `search_documents` (which called
`await asyncio.to_thread(_search_blocking, query)` directly, with no
existence check and no try/except) and all three failed with an unhandled
exception escaping the `await`: a `RuntimeError` out of FAISS's own C++ layer
for the missing-index case ("could not open ... for reading"), the same
exception type with a different message for the corrupt-index case ("not
recognized"), and the injected simulated-429 `RuntimeError` for the
transient-failure case - confirming this suite catches the exact defect
finding 2 describes rather than merely matching the shape of the new code.
"""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document

from src import config
from src.embedding.embedder import GeminiEmbeddings
from src.retrieval import faiss_store
from src.tools import document_search

BUILD_COMMAND_PHRASE = "python -m src.dataset"
REPORT_GAP_PHRASE = "report_gap"

# Mutually exclusive by construction - the missing-index message must claim
# retrying is futile, the "index is present but something else failed"
# message must claim the opposite. Checking for the right positive claim (and
# the absence of the other one) is a stronger assertion than banning a single
# word, which a differently-worded wrong diagnosis could still slip past.
RETRY_WONT_HELP_PHRASE = "retrying will not help"
RETRY_MAY_HELP_PHRASE = "retrying once may succeed"


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """`_index` is a process-wide `lru_cache(maxsize=1)` - stop it leaking across tests.

    Without this, whichever test happens to load an index first would poison
    every test after it: a successful load caches forever (only exceptions are
    never cached, by `functools.lru_cache`'s own contract), and the fixture
    below points `config.FAISS_INDEX_DIR` at a different tmp_path per test.
    """
    document_search._index.cache_clear()
    yield
    document_search._index.cache_clear()


def _build_and_save_tiny_index(directory) -> None:
    """One real chunk through the real build/save path, written into `directory`."""
    dim = 8
    unit_vector = np.zeros(dim, dtype="float32")
    unit_vector[0] = 1.0  # already unit-norm, satisfies build_index's own check
    chunk = Document(
        page_content="Net income for FY2024 was $1.2 billion.",
        metadata={"source": "ifc-2024-annual-report.pdf"},
    )
    index = faiss_store.build_index([chunk], [unit_vector.tolist()])
    faiss_store.save_index(index)


@pytest.mark.asyncio
async def test_missing_index_degrades_with_the_build_command(tmp_path, monkeypatch) -> None:
    """The audit's exact live scenario: no FAISS files on disk at all."""
    monkeypatch.setattr(config, "FAISS_INDEX_DIR", tmp_path / "does-not-exist")

    result = await document_search.search_documents(
        "IFC's FY24 net income", "What was FY2024 net income?"
    )

    assert isinstance(result, list) and len(result) == 1, f"expected a single-item list, got {result!r}"
    assert "error" in result[0], f"expected an 'error' key, got {result[0]!r}"
    message = result[0]["error"]
    assert BUILD_COMMAND_PHRASE in message, f"missing the build command: {message!r}"
    assert REPORT_GAP_PHRASE in message, f"missing the report_gap steer: {message!r}"
    # The wrong diagnosis for this shape: claiming a setup problem might
    # clear on its own would send the model back into the same dead end.
    assert RETRY_WONT_HELP_PHRASE in message, f"missing the futile-retry claim: {message!r}"
    assert RETRY_MAY_HELP_PHRASE not in message, f"wrongly hinted this might clear on retry: {message!r}"


@pytest.mark.asyncio
async def test_transient_failure_degrades_without_suggesting_a_rebuild(tmp_path, monkeypatch) -> None:
    """A present, valid index that fails later - the transient-Vertex-error shape.

    Builds a real index so it genuinely loads, then simulates the audit's
    named failure (a 429/503 inside `GeminiEmbeddings.embed_query`, which
    `FAISS.similarity_search` calls to embed the query before it can search)
    by patching only that one method.
    """
    monkeypatch.setattr(config, "FAISS_INDEX_DIR", tmp_path)
    _build_and_save_tiny_index(tmp_path)

    def _simulated_vertex_429(self, text):
        raise RuntimeError("429 Resource has been exhausted (simulated Vertex quota error)")

    monkeypatch.setattr(GeminiEmbeddings, "embed_query", _simulated_vertex_429)

    result = await document_search.search_documents(
        "IFC's FY24 net income", "What was FY2024 net income?"
    )

    assert isinstance(result, list) and len(result) == 1, f"expected a single-item list, got {result!r}"
    assert "error" in result[0], f"expected an 'error' key, got {result[0]!r}"
    message = result[0]["error"]
    assert REPORT_GAP_PHRASE in message, f"missing the report_gap steer: {message!r}"
    # The wrong diagnosis for this shape: the index is present and valid, so
    # claiming retrying is futile - or naming the build command at all - would
    # misdiagnose a live blip as a missing build step, the opposite mistake
    # to the one this fix targets.
    assert RETRY_MAY_HELP_PHRASE in message, f"missing the may-clear-on-retry claim: {message!r}"
    assert RETRY_WONT_HELP_PHRASE not in message, f"wrongly claimed this cannot clear on retry: {message!r}"
    assert BUILD_COMMAND_PHRASE not in message, f"wrongly suggested rebuilding a present, valid index: {message!r}"
    assert "429" in message, f"expected the underlying failure detail to surface: {message!r}"


@pytest.mark.asyncio
async def test_corrupt_index_is_not_mistaken_for_a_missing_one(tmp_path, monkeypatch) -> None:
    """Files present but not a valid index - must land in the "other failure" bucket, not "missing".

    This is the exact case `_index_files_present`'s docstring argues an
    existence check handles correctly and message-matching on FAISS's
    `RuntimeError` would not: a missing `.faiss` and a corrupt-but-present
    `.faiss` raise the same exception type with different text (verified
    directly against the installed `faiss` package - "could not open ... for
    reading" versus "not recognized"). Without an existence check, whoever
    wrote a text-matching discriminator could plausibly get this one wrong;
    this test is what would catch that regression.
    """
    monkeypatch.setattr(config, "FAISS_INDEX_DIR", tmp_path)
    (tmp_path / f"{faiss_store.INDEX_NAME}.faiss").write_bytes(b"not a real faiss index")
    (tmp_path / f"{faiss_store.INDEX_NAME}.pkl").write_bytes(b"not a real pickle either")

    result = await document_search.search_documents(
        "IFC's FY24 net income", "What was FY2024 net income?"
    )

    assert isinstance(result, list) and len(result) == 1, f"expected a single-item list, got {result!r}"
    assert "error" in result[0], f"expected an 'error' key, got {result[0]!r}"
    message = result[0]["error"]
    assert REPORT_GAP_PHRASE in message, f"missing the report_gap steer: {message!r}"
    assert RETRY_MAY_HELP_PHRASE in message, f"missing the may-clear-on-retry claim: {message!r}"
    assert RETRY_WONT_HELP_PHRASE not in message, f"wrongly claimed this cannot clear on retry: {message!r}"
    assert BUILD_COMMAND_PHRASE not in message, (
        f"corrupt-but-present index wrongly diagnosed as missing: {message!r}"
    )
