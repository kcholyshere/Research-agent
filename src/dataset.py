"""Build the knowledge-base index: chunk every file under data/raw/, embed once,
save the FAISS index. Run as `python -m src.dataset` after dropping the phase 1
corpus into data/raw/.

Adapted from Finrag's dataset.py (single-embed pattern, audit A16 there).
"""

from src import config
from src.ingestion.chunk import chunk_files, save_chunks
from src.embedding.embedder import GeminiEmbeddings
from src.retrieval import faiss_store

SOURCE_GLOBS = ("*.txt", "*.md")


def run() -> None:
    paths = sorted(p for pattern in SOURCE_GLOBS for p in config.RAW_DATA_DIR.glob(pattern))
    if not paths:
        raise SystemExit(f"No source documents found under {config.RAW_DATA_DIR} ({SOURCE_GLOBS})")

    chunks = chunk_files(paths)
    save_chunks(chunks)
    print(f"{len(paths)} files -> {len(chunks)} chunks")

    print(f"Embedding {len(chunks)} chunks...")
    vectors = GeminiEmbeddings().embed_documents([chunk.page_content for chunk in chunks])

    index = faiss_store.build_index(chunks, vectors)
    faiss_store.save_index(index)
    print("FAISS index built and saved")


if __name__ == "__main__":
    run()
