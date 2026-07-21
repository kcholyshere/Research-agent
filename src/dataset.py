"""Build the knowledge-base index: chunk every file under data/raw/, embed once,
save the FAISS index. Run as `python -m src.dataset` after dropping the phase 1
corpus into data/raw/.

Adapted from Finrag's dataset.py (single-embed pattern, audit A16 there).
"""

import os

# faiss-cpu and torch (pulled in by docling for PDF parsing, ADR-0001) each
# bundle their own copy of libomp.dylib on macOS (confirmed distinct binaries
# under .venv/lib/python*/site-packages/{torch/lib,faiss/.dylibs}). Loading
# both in one process aborts with "OMP: Error #15: Initializing libomp.dylib,
# but found libomp.dylib already initialized" unless the duplicate-load check
# is disabled - but disabling that check only silences the abort message; it
# does not stop the two runtimes from each spinning up their own worker
# thread pool. If either one actually goes parallel (FAISS's HNSW index build
# does, by default using all cores), the two pools corrupt each other's
# thread-suspend/barrier state and the process SIGSEGVs inside libomp
# (confirmed via a macOS crash report: EXC_BAD_ACCESS in
# __kmp_suspend_initialize_thread / __kmp_fork_barrier) - silent, no Python
# traceback, since it's a native crash below the interpreter. Forcing both
# runtimes to a single thread means neither ever enters a parallel region, so
# there is no worker pool to corrupt; two OpenMP runtimes can then coexist
# (just not concurrently execute) in one process. This is the only entrypoint
# that imports both libraries in one process (the agent never imports torch
# at runtime), so both workarounds are scoped here rather than set globally.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src import config
from src.ingestion.chunk import chunk_files, chunk_pdf, save_chunks
from src.embedding.embedder import GeminiEmbeddings
from src.retrieval import faiss_store

SOURCE_GLOBS = ("*.txt", "*.md", "*.pdf")


def run() -> None:
    paths = sorted(p for pattern in SOURCE_GLOBS for p in config.RAW_DATA_DIR.glob(pattern))
    if not paths:
        raise SystemExit(f"No source documents found under {config.RAW_DATA_DIR} ({SOURCE_GLOBS})")

    pdf_paths = [p for p in paths if p.suffix == ".pdf"]
    text_paths = [p for p in paths if p.suffix != ".pdf"]

    chunks = chunk_files(text_paths) if text_paths else []
    for pdf_path in pdf_paths:
        chunks += chunk_pdf(pdf_path)
    save_chunks(chunks)
    print(f"{len(paths)} files -> {len(chunks)} chunks")

    print(f"Embedding {len(chunks)} chunks...")
    vectors = GeminiEmbeddings().embed_documents([chunk.page_content for chunk in chunks])

    index = faiss_store.build_index(chunks, vectors)
    faiss_store.save_index(index)
    print("FAISS index built and saved")


if __name__ == "__main__":
    run()
