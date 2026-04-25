"""
Configuration module.
Loads environment variables from .env and exposes them as constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Required environment variables ---

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

POSTGRES_URL = os.getenv("POSTGRES_URL")
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL is not set. Please add it to your .env file.")

# --- Embedding settings ---

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072  # gemini-embedding-001 output dimension

# --- Chunking settings ---

CHUNK_SENTENCES = 6   # number of sentences per chunk
CHUNK_OVERLAP = 2     # overlapping sentences between consecutive chunks
