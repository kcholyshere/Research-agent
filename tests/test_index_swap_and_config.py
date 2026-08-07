"""Regression tests for audit findings 14b and 15a.

## 14b - the .faiss/.pkl swap in save_index()/load_index()

`save_index()`'s own docstring used to claim the two-file move it does was
"atomic from the caller's view". It is not: each `os.replace()` is atomic
individually, but the pair is not swapped as one operation, so a crash
between the two calls can leave a fresh `.faiss` beside a stale `.pkl` (or
the reverse). The comment has been corrected to name that window plainly
(see `src/retrieval/faiss_store.py`), and `load_index()` now checks that the
loaded index and docstore agree on vector count, raising loudly instead of
either a bare `KeyError` deep inside a later search or a silent, wrong
result set.

What is deliberately NOT tested here: the same-count-different-content case
the audit named as the truly silent one (a rebuild that lands on the same
chunk count with different text - every id still resolves, just to the
wrong chunk). Nothing in this file can detect that without a content hash
tying the two files together, which was not implemented - see
`save_index()`'s docstring for why. The count-mismatch test below is the
only case a load-time check can cheaply catch.

The corruption is simulated by hand-editing the saved `.pkl` after a real
`save_index()` call, rather than by actually killing a process mid-swap -
that is the only way to get a deterministic, offline repro of "the pair
disagrees on count" without relying on timing.

## 15a - a blank GOOGLE_CLOUD_PROJECT

Verified against the installed `google.genai._api_client.BaseApiClient`:
`self.project = project or env_project` and, if still falsy, `load_auth()`
resolves a project from Application Default Credentials rather than
raising. So a blank `.env` does not fail - it silently sends every Vertex
call to whichever project ADC defaults to (the exact misdiagnosis
ADR-0004 spent real time on).

The fix lives in `src/services/genai_client.get_client()`, not
`src/config.py` at import time, because `config.py` is imported
unconditionally - including by this offline suite - by code that never
touches Vertex. The tests below check both ends of that decision: the
error is loud and named at `get_client()`, and importing `src.config` with
a blank project still succeeds.
"""

from __future__ import annotations

import importlib
import pickle

import pytest
from langchain_core.documents import Document

from src import config
from src.retrieval import faiss_store
from src.services import genai_client

# Exactly unit-norm (a standard basis vector), so build_index()'s own
# unit-norm check (UNIT_NORM_ATOL) never rejects this fixture data - the
# point of these tests is the swap/count logic, not the norm check.
_DIM = 4


def _unit_vector(i: int) -> list[float]:
    vec = [0.0] * _DIM
    vec[i] = 1.0
    return vec


def _make_chunks_and_vectors(n: int) -> tuple[list[Document], list[list[float]]]:
    chunks = [
        Document(page_content=f"chunk {i}", metadata={"source": "test.pdf"})
        for i in range(n)
    ]
    vectors = [_unit_vector(i % _DIM) for i in range(n)]
    return chunks, vectors


def test_save_then_load_round_trips_when_the_pair_is_consistent(tmp_path, monkeypatch):
    """Sanity check: an uninterrupted save/load still works (no false positive)."""
    monkeypatch.setattr(config, "FAISS_INDEX_DIR", tmp_path)
    chunks, vectors = _make_chunks_and_vectors(3)
    index = faiss_store.build_index(chunks, vectors)

    faiss_store.save_index(index)
    loaded = faiss_store.load_index()

    assert loaded.index.ntotal == 3
    assert len(loaded.index_to_docstore_id) == 3


def test_load_index_raises_loudly_on_count_mismatch(tmp_path, monkeypatch):
    """The interrupted-swap case: .faiss and .pkl disagree on vector count.

    Simulates a crash between save_index()'s two os.replace() calls by
    hand-editing the saved .pkl afterwards to claim one more id than the
    .faiss file actually holds vectors for - the same shape of desync a
    stale .pkl left behind by an interrupted rebuild would produce.
    """
    monkeypatch.setattr(config, "FAISS_INDEX_DIR", tmp_path)
    chunks, vectors = _make_chunks_and_vectors(3)
    index = faiss_store.build_index(chunks, vectors)
    faiss_store.save_index(index)

    pkl_path = tmp_path / f"{faiss_store.INDEX_NAME}.pkl"
    with open(pkl_path, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # The .faiss file on disk still has 3 vectors; make the .pkl claim 4 ids,
    # as if an old, pre-rebuild .pkl had briefly survived a crash mid-swap.
    docstore._dict["3"] = Document(page_content="orphaned chunk", metadata={})
    index_to_docstore_id[3] = "3"
    with open(pkl_path, "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f)

    with pytest.raises(ValueError, match="size mismatch"):
        faiss_store.load_index()


def test_get_client_raises_a_named_error_when_project_is_blank(monkeypatch):
    """A blank GOOGLE_CLOUD_PROJECT must fail loudly at get_client(), not
    silently fall through to whatever project ADC defaults to."""
    monkeypatch.setattr(config, "GCP_PROJECT", "")
    genai_client.get_client.cache_clear()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            genai_client.get_client()
        message = str(exc_info.value)
        assert "GOOGLE_CLOUD_PROJECT" in message
        assert ".env" in message
    finally:
        genai_client.get_client.cache_clear()


def test_get_client_succeeds_once_project_is_set(monkeypatch):
    """A real (or real-looking) project builds a client with no network call.

    Constructing genai.Client(vertexai=True, project=..., location=...)
    with a truthy project never reaches Application Default Credentials
    (verified against the installed google.genai._api_client.py: the
    load_auth() call is only made when `not self.project`), so this is
    genuinely offline - it also demonstrates that lru_cache does not cache
    the RuntimeError from the previous test, so a fixed environment recovers
    on the very next call.
    """
    monkeypatch.setattr(config, "GCP_PROJECT", "fake-test-project")
    genai_client.get_client.cache_clear()
    try:
        client = genai_client.get_client()
        assert client._api_client.project == "fake-test-project"
    finally:
        genai_client.get_client.cache_clear()


def test_importing_config_survives_a_blank_project(monkeypatch):
    """config.py must not raise at import time on a blank/missing project -
    the offline suite (and any module that never touches Vertex) imports it
    unconditionally, per this test file's module docstring."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "")
    try:
        importlib.reload(config)
        assert config.GCP_PROJECT == ""
    finally:
        # Restore the real environment's value before other tests import
        # (or re-import) config - monkeypatch only undoes the env var when
        # this test function returns, so the module itself needs reloading
        # again here to pick that restoration up within this test's scope.
        monkeypatch.undo()
        importlib.reload(config)
