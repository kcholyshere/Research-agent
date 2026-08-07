import sys

from langchain_core.embeddings import Embeddings

from google.genai import types

from src import config
from src.services.genai_client import get_client

BATCH_SIZE = 100


def _report_truncation(total: int, truncated: list[int], unknown: list[int]) -> None:
    """Print a build-time report of embedding truncation (audit finding 10).

    `embed_content`'s per-embedding `statistics.truncated` is the only signal
    that a chunk's vector was built from a cut-down prefix rather than its
    full text - table chunks have no size cap (chunk.py deliberately never
    splits a table), so a large table can silently exceed the model's input
    limit. The stored page_content is unaffected either way, which is what
    makes this failure mode invisible without a check like this one: the
    chunk still "exists" and reads correctly if you open the docstore, it
    just becomes unfindable by similarity search for anything in its cut
    tail.

    `statistics` itself is documented in the installed google-genai package
    as "Gemini Enterprise Agent Platform only" (types.py, ContentEmbedding
    and ContentEmbeddingStatistics), so on plain Vertex AI it may come back
    as None for every chunk - i.e. the check may never be able to confirm
    "not truncated", only "unknown". That is reported as its own bucket
    rather than folded into "not truncated", so a maintainer never reads
    silence as an all-clear when it is actually "couldn't check".

    Printed to stderr (matching dataset.py's print-based progress output,
    which has no logging configuration to hook into) so the report survives
    stdout being redirected or piped, and always emitted - even the all-clear
    case - so its absence is never mistaken for "the check didn't run".
    """
    print("-" * 70, file=sys.stderr)
    if truncated:
        print(
            f"EMBEDDING TRUNCATION: {len(truncated)}/{total} chunks were "
            "truncated by the embedding API before being embedded (input "
            "longer than the model's max input length). Each is unfindable "
            "by similarity search for anything past its truncated point - "
            "see agent_docs/audit.md finding 10.",
            file=sys.stderr,
        )
        print(f"  Truncated chunk indices: {truncated}", file=sys.stderr)
    else:
        print(
            f"EMBEDDING TRUNCATION: 0/{total} chunks confirmed truncated.",
            file=sys.stderr,
        )
    if unknown:
        print(
            f"EMBEDDING TRUNCATION STATUS UNKNOWN for {len(unknown)}/{total} "
            "chunks: the embed_content response carried no `statistics` for "
            "them, so truncation could not be checked (this field is "
            "documented Gemini Enterprise Agent Platform only, and may be "
            "structurally absent on this deployment). Treat these as "
            "unconfirmed, not as verified untruncated.",
            file=sys.stderr,
        )
    print("-" * 70, file=sys.stderr)


class GeminiEmbeddings(Embeddings):
    """LangChain Embeddings adapter around the Vertex AI gemini-embedding-001 model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = get_client()
        vectors: list[list[float]] = []
        truncated_indices: list[int] = []
        unknown_indices: list[int] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            for offset, e in enumerate(resp.embeddings):
                if e.statistics is None:
                    unknown_indices.append(i + offset)
                elif e.statistics.truncated:
                    truncated_indices.append(i + offset)
            vectors.extend(e.values for e in resp.embeddings)
        _report_truncation(len(texts), truncated_indices, unknown_indices)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        client = get_client()
        resp = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return resp.embeddings[0].values
