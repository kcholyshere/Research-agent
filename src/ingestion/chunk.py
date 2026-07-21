"""Chunking for the phase 1 knowledge base.

Adapted from Finrag's chunk.py: same splitter settings and JSONL persistence
idiom, minus the Finrag-specific image enrichment path. Plain text/markdown
files are split as-is; PDFs go through Docling parsing (parse.py) first, then
section-aware text chunking plus one chunk per table (ADR-0001).
"""

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.ingestion.parse import extract_table_records, extract_text_records, load_or_parse_pdf

NOISE_LABELS = {"page_footer", "page_header"}

CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"


def chunk_files(paths: list[Path]) -> list[Document]:
    """Split each plain text/markdown source file into overlapping chunks tagged with its origin."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    texts = [path.read_text() for path in paths]
    metadatas = [{"source": str(path.relative_to(config.PROJECT_ROOT))} for path in paths]
    return splitter.create_documents(texts, metadatas=metadatas)


def _group_into_sections(records: list[dict]) -> list[dict]:
    """Merge consecutive text records sharing a section into one block per section."""
    sections: list[dict] = []
    current = None

    for record in records:
        if record["label"] in NOISE_LABELS:
            continue

        section = record["section"]
        if current is None or current["section"] != section:
            current = {"section": section, "start_page": record["page"], "text": record["text"]}
            sections.append(current)
        else:
            current["text"] += "\n" + record["text"]
            # A record with no page (e.g. an item Docling couldn't place) must
            # never overwrite an already-known end_page with None.
            if record["page"] is not None:
                current["end_page"] = record["page"]

    for section in sections:
        section.setdefault("end_page", section["start_page"])

    return sections


def chunk_pdf(pdf_path: Path) -> list[Document]:
    """Parse a PDF (Docling) and split it into section-aware text chunks plus
    one chunk per table - a table is already a coherent unit, so it is kept
    whole rather than recursively split."""
    document = load_or_parse_pdf(pdf_path)
    source = str(pdf_path.relative_to(config.PROJECT_ROOT))

    sections = _group_into_sections(extract_text_records(document))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    text_chunks = splitter.create_documents(
        [s["text"] for s in sections],
        metadatas=[
            {
                "source": source,
                "section": s["section"],
                "start_page": s["start_page"],
                "end_page": s["end_page"],
            }
            for s in sections
        ],
    )

    table_chunks = [
        Document(
            page_content="\n\n".join(
                filter(None, [f"Table: {record['caption'] or record['section']}", record["text"]])
            ),
            metadata={
                "source": source,
                "section": record["section"],
                "start_page": record["page"],
                "end_page": record["page"],
            },
        )
        for record in extract_table_records(document)
    ]

    return text_chunks + table_chunks


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
