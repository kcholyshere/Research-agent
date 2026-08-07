import os
import tempfile

import faiss
import numpy as np

# langchain_community is sunset upstream, but there is no drop-in replacement:
# langchain_classic.vectorstores.FAISS / .docstore.in_memory.InMemoryDocstore
# are themselves just deprecated lazy re-exports of these same community
# classes (confirmed against the installed package), so routing through them
# would add a deprecation warning without removing the dependency. Pinned
# deliberately until upstream gives FAISS a real non-community home.
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src import config
from src.embedding.embedder import GeminiEmbeddings

INDEX_NAME = "knowledge_base"

# HNSW gives FAISS an O(log n) graph-search profile, instead of the flat/
# brute-force O(n) scan `FAISS.from_documents` builds by default.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# FAISS's IndexHNSWFlat scores L2 distance, not cosine similarity - the
# ranking only matches cosine similarity while every vector is unit-norm (L2
# on unit vectors is a monotonic function of cosine similarity). Nothing
# upstream enforces that, so a future embedding model or config change that
# returns unnormalised vectors would silently produce a wrong ranking instead
# of erroring, hence this check.
UNIT_NORM_ATOL = 1e-3


def build_index(chunks: list[Document], vectors: list[list[float]]) -> FAISS:
    """Build the HNSW index from chunks and their precomputed embedding vectors."""
    if len(vectors) != len(chunks):
        raise ValueError(f"Got {len(vectors)} vectors for {len(chunks)} chunks")

    norms = np.linalg.norm(np.array(vectors, dtype="float32"), axis=1)
    if not np.allclose(norms, 1.0, atol=UNIT_NORM_ATOL):
        raise ValueError(
            "Embedding vectors are not unit-norm (max deviation "
            f"{np.max(np.abs(norms - 1.0)):.4f}). FAISS's L2 index only ranks "
            "identically to cosine similarity when vectors are unit-norm - "
            "an unnormalised embedding model would silently produce a wrong "
            "ranking instead of erroring, hence this check."
        )

    embeddings = GeminiEmbeddings()  # kept on the store for query-time embed_query

    index = faiss.IndexHNSWFlat(len(vectors[0]), HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(np.array(vectors, dtype="float32"))

    docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(chunks)})
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    return FAISS(embeddings, index, docstore, index_to_docstore_id)


def save_index(index: FAISS) -> None:
    """Write the index to a temp directory, then move both files into place.

    save_local() writes the .faiss and .pkl files as two separate calls; an
    interrupt between them would leave the pair desynced (one stale, one
    fresh) with nothing to detect it. Building in a temp dir first means both
    files are complete and mutually consistent before either os.replace()
    below runs, and each os.replace() is individually atomic - the
    destination path never shows a partially-written file.

    That is weaker than "the swap is atomic as a pair", which this used to
    claim and does not hold: a crash or kill between the two os.replace()
    calls leaves whichever suffix was replaced first fresh and the other
    stale (audit finding 14b). A directory-level swap - build the whole pair
    under one new directory, then a single os.replace() of that directory -
    would close the window, but the destination already holds the previous
    pair from the last build, and os.replace() refuses to replace a
    non-empty directory (confirmed on this platform: ENOTEMPTY). Doing this
    properly needs an extra layer of indirection - two candidate directories
    plus a symlink flipped atomically between them - which is a bigger
    restructure than this reused, treated-as-proven file warrants for one
    finding; recorded here as the fix if the window below ever bites.

    If the crash lands in that window and it changes the vector COUNT,
    load_index() below now catches it and raises loudly. It cannot catch a
    rebuild that lands on the same chunk count with different content - every
    id would still resolve, just to the wrong chunk text, and nothing short
    of a content hash tying the two files together (stored alongside the
    pickle, checked on load) would detect that. That hash is not implemented;
    a same-count desync from an interrupted save is a live, silent risk.
    """
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.FAISS_INDEX_DIR) as tmp_dir:
        index.save_local(tmp_dir, index_name=INDEX_NAME)
        for suffix in (".faiss", ".pkl"):
            os.replace(
                os.path.join(tmp_dir, INDEX_NAME + suffix),
                config.FAISS_INDEX_DIR / (INDEX_NAME + suffix),
            )


def load_index() -> FAISS:
    """Load the index, then check the .faiss/.pkl pair actually agree on size.

    save_index() above is not atomic as a pair (see its docstring): a crash
    between its two os.replace() calls can leave a fresh .faiss beside a
    stale .pkl or vice versa. Verified against the installed
    langchain_community.vectorstores.faiss.FAISS.load_local/__init__: neither
    compares index.ntotal to len(index_to_docstore_id), so FAISS itself does
    not catch a count mismatch here - it would instead surface later,
    unpredictably, as a bare KeyError deep inside similarity_search if the
    index ends up with MORE vectors than the docstore has ids, or as silently
    fewer/no results if it has FEWER, never as a clear error at load time.
    The check below makes that count-mismatch case loud at the one place that
    can name it clearly, rather than leaving it as a mystery either at query
    time or never.

    It does not, and cannot cheaply, catch a rebuild that kept the same
    chunk count but changed the content - every id would still resolve, just
    to the wrong text. See save_index()'s docstring for why that residual
    case is left as a named risk rather than fixed here.
    """
    loaded = FAISS.load_local(
        str(config.FAISS_INDEX_DIR),
        GeminiEmbeddings(),
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    if loaded.index.ntotal != len(loaded.index_to_docstore_id):
        raise ValueError(
            f"Index/docstore size mismatch in {config.FAISS_INDEX_DIR}: "
            f"{INDEX_NAME}.faiss has {loaded.index.ntotal} vectors but "
            f"{INDEX_NAME}.pkl has {len(loaded.index_to_docstore_id)} ids. "
            "This is the failure mode save_index()'s docstring names: a "
            "crash between its two os.replace() calls left the pair "
            "desynced. Rebuild the index (python -m src.dataset) from a "
            "known-good corpus rather than trusting either file alone."
        )
    return loaded
