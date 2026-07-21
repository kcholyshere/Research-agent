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

# Models (phase 1 requirements specify Gemini 2.0 Flash; retired from
# gd-gcp-internship-ds's Vertex AI catalogue by the time of verification -
# see ADR-0004 for the substitution)
GEMINI_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Knowledge base source documents (phase 1 corpus: IFC's 2024 annual report,
# see ADR-0001 - same file Finrag used, dropped as-is into data/raw/)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Chunking (carried over from Finrag's tuned values)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Vector store (phase 1 requirement: FAISS, in-memory document search)
FAISS_INDEX_DIR = PROJECT_ROOT / "models" / "faiss"

# Retrieval
TOP_K = 4
