"""
Database module.
Handles PostgreSQL connection and storage of document chunks with pgvector.
"""

from sqlalchemy import create_engine, Column, Integer, Text, String, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from config import POSTGRES_URL, EMBEDDING_DIMENSION

Base = declarative_base()


class DocumentChunk(Base):
    """SQLAlchemy model for storing document chunks with their embeddings."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=False)  # pgvector
    filename = Column(Text, nullable=False)
    split_strategy = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


# Database engine and session factory (created once, reused)
_engine = create_engine(POSTGRES_URL)
_SessionLocal = sessionmaker(bind=_engine)


def init_db() -> None:
    """
    Initialize the database:
    1. Enable the pgvector extension
    2. Create all tables if they don't exist

    Safe to call multiple times (idempotent).
    """
    with _engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(_engine)
    print("Database initialized successfully.")


def store_chunks(
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
    strategy: str,
) -> int:
    """
    Store document chunks and their embeddings in the database.

    Embeddings are stored as VECTOR(3072) using pgvector.
    All chunks are inserted in a single transaction for atomicity.

    Args:
        filename: Source document name.
        chunks: List of text chunks.
        embeddings: List of embedding vectors (must match chunks length).
        strategy: Name of the chunking strategy used.

    Returns:
        Number of rows inserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
        )

    session = _SessionLocal()
    try:
        records = [
            DocumentChunk(
                chunk_text=chunk,
                embedding=embedding,
                filename=filename,
                split_strategy=strategy,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        session.add_all(records)
        session.commit()
        return len(records)
    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Failed to store chunks in database: {e}") from e
    finally:
        session.close()
