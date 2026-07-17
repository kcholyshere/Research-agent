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

INDEX_NAME = "ifc_annual_report_2024"

# HNSW gives FAISS the same O(log n) graph-search profile as Qdrant's default
# index, instead of the flat/brute-force O(n) scan `FAISS.from_documents` builds.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# FAISS's IndexHNSWFlat scores L2 distance, Qdrant's collection scores cosine -
# the two rankings only agree while every vector is unit-norm (L2 on unit
# vectors is a monotonic function of cosine similarity). Nothing upstream
# enforces that, so a future embedding model or config change that returns
# unnormalised vectors would silently desync FAISS ranking from Qdrant's
# without either side erroring.
UNIT_NORM_ATOL = 1e-3


def build_index(chunks: list[Document], vectors: list[list[float]]) -> FAISS:
    """Build the HNSW index from chunks and their precomputed embedding vectors.

    Vectors are computed once by the caller (dataset.py) and shared with the
    Qdrant build (A16) - embedding all ~852 chunks is the single most
    expensive/costly step in a rebuild, so doing it twice (once per store) was
    pure waste and a drift path between the two indexes.
    """
    if len(vectors) != len(chunks):
        raise ValueError(f"Got {len(vectors)} vectors for {len(chunks)} chunks")

    norms = np.linalg.norm(np.array(vectors, dtype="float32"), axis=1)
    if not np.allclose(norms, 1.0, atol=UNIT_NORM_ATOL):
        raise ValueError(
            "Embedding vectors are not unit-norm (max deviation "
            f"{np.max(np.abs(norms - 1.0)):.4f}). FAISS's L2 index and Qdrant's "
            "cosine index only rank identically when vectors are unit-norm - "
            "an unnormalised embedding model would silently desync the two "
            "stores' rankings instead of erroring, hence this check."
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
    fresh) with nothing to detect it. Building in a temp dir and moving the
    finished pair in as a batch keeps the swap atomic from the caller's view.
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
    return FAISS.load_local(
        str(config.FAISS_INDEX_DIR),
        GeminiEmbeddings(),
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
