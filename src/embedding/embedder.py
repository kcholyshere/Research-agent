from langchain_core.embeddings import Embeddings

from google.genai import types

from src import config
from src.services.genai_client import get_client

BATCH_SIZE = 100


class GeminiEmbeddings(Embeddings):
    """LangChain Embeddings adapter around the Vertex AI gemini-embedding-001 model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = get_client()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend(e.values for e in resp.embeddings)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        client = get_client()
        resp = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return resp.embeddings[0].values
