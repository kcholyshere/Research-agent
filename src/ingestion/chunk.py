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
                # Symmetric guard for start_page (finding 15c): if the
                # section's opening record(s) lacked Docling provenance,
                # start_page was seeded as None above and would otherwise
                # stay None forever even once a later record in the same
                # section supplies a real page - producing a citation like
                # "page None-42" instead of "page 42-42". Backfill it from
                # the first record that does carry a page, same as end_page.
                if current["start_page"] is None:
                    current["start_page"] = record["page"]

    for section in sections:
        section.setdefault("end_page", section["start_page"])

    return sections


def _split_table_text(text: str, max_chars: int) -> list[str]:
    """Split an oversized markdown table into parts that each keep the header.

    Splitting a table on raw character count would be worse than not splitting
    it: every part after the first would be a run of bare `| value | value |`
    rows with no column names, which is less retrievable than the truncated
    original rather than more. So the split is on ROWS, and the header block
    (everything up to and including the `|---|` separator) is repeated at the
    top of every part, which is what makes each part independently meaningful
    to both an embedding and a reader.

    A row longer than `max_chars` on its own is emitted as its own oversized
    part rather than being cut mid-row. That trades a rare truncation for
    never emitting a malformed row, and no row in the current corpus is
    anywhere near it - the widest is under 400 characters.

    Text with no separator row is not a markdown table at all (Docling
    occasionally emits a bare block), so it falls back to plain slicing.
    """
    lines = text.split("\n")
    separator = next(
        (i for i, line in enumerate(lines) if "-" in line and set(line.strip()) <= set("|-: ")),
        None,
    )
    if separator is None:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [text]

    header = lines[: separator + 1]
    header_chars = sum(len(line) + 1 for line in header)
    parts: list[str] = []
    current: list[str] = []
    current_chars = 0
    for row in lines[separator + 1 :]:
        row_chars = len(row) + 1
        if current and header_chars + current_chars + row_chars > max_chars:
            parts.append("\n".join(header + current))
            current, current_chars = [], 0
        current.append(row)
        current_chars += row_chars
    if current:
        parts.append("\n".join(header + current))
    return parts or [text]


def chunk_pdf(pdf_path: Path) -> list[Document]:
    """Parse a PDF (Docling) and split it into section-aware text chunks plus
    table chunks.

    A table is kept whole where it fits, because it is already a coherent
    unit and recursive character splitting would cut it mid-row. Where it does
    not fit it is split by rows with the header repeated - see
    `_split_table_text` and `config.TABLE_CHUNK_MAX_CHARS` for why keeping
    every table whole turned out to be silently lossy."""
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

    table_chunks: list[Document] = []
    for record in extract_table_records(document):
        caption = f"Table: {record['caption'] or record['section']}"
        parts = _split_table_text(record["text"], config.TABLE_CHUNK_MAX_CHARS)
        for index, part in enumerate(parts, start=1):
            # The caption is repeated on every part alongside the header row,
            # for the same reason the header is: a part that does not say
            # which table it belongs to is far harder to retrieve, and the
            # part number is what stops a reader thinking they have the whole
            # table when they have a third of it.
            heading = caption if len(parts) == 1 else f"{caption} (part {index} of {len(parts)})"
            table_chunks.append(
                Document(
                    page_content="\n\n".join(filter(None, [heading, part])),
                    metadata={
                        "source": source,
                        "section": record["section"],
                        "start_page": record["page"],
                        "end_page": record["page"],
                        "table_part": index,
                        "table_parts": len(parts),
                    },
                )
            )

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
