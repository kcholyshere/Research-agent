"""Chunking for the phase 1 knowledge base.

Adapted from Finrag's chunk.py: same splitter settings and JSONL persistence
idiom, minus the Finrag-specific table/image enrichment paths. The knowledge
base for this project is plain text/markdown files under data/raw/ until the
phase 1 corpus is decided.
"""

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"


def chunk_files(paths: list[Path]) -> list[Document]:
    """Split each source file into overlapping chunks tagged with its origin."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    texts = [path.read_text() for path in paths]
    metadatas = [{"source": str(path.relative_to(config.PROJECT_ROOT))} for path in paths]
    return splitter.create_documents(texts, metadatas=metadatas)


def save_chunks(chunks: list[Document]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        for i, chunk in enumerate(chunks):
            record = {"id": i, "text": chunk.page_content, **chunk.metadata}
            f.write(json.dumps(record) + "\n")


@lru_cache(maxsize=1)
def load_chunks() -> list[Document]:
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            record = json.loads(line)
            text = record.pop("text")
            record.pop("id", None)
            chunks.append(Document(page_content=text, metadata=record))
    return chunks
