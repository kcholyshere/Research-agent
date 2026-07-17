import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vertex AI auth (no API keys - relies on Application Default Credentials).
# ADK reads GOOGLE_GENAI_USE_VERTEXAI/GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION
# from the environment itself; these mirrors are for our own modules.
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# Models (per phase 1 requirements: Gemini 2.0 Flash via Vertex AI)
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Knowledge base source documents (phase 1: private knowledge base, TBD)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Chunking (carried over from Finrag's tuned values)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Vector store (phase 1 requirement: FAISS, in-memory document search)
FAISS_INDEX_DIR = PROJECT_ROOT / "models" / "faiss"

# Retrieval
TOP_K = 4
