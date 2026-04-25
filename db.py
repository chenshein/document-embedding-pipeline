"""
Database module.
Handles PostgreSQL connection and storage of document chunks.
Embeddings are stored as PostgreSQL FLOAT arrays (vector representation).
"""

from sqlalchemy import create_engine, Column, Integer, Text, String, DateTime, ARRAY, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

from config import POSTGRES_URL

Base = declarative_base()


class DocumentChunk(Base):
    """SQLAlchemy model for storing document chunks with their embeddings."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)  # vector embedding
    filename = Column(String(255), nullable=False)
    split_strategy = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


# Database engine and session factory (created once, reused)
_engine = create_engine(POSTGRES_URL)
_SessionLocal = sessionmaker(bind=_engine)


def init_db() -> None:
    """
    Initialize the database: create all tables if they don't exist.
    Safe to call multiple times (idempotent).
    """
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

    Embeddings are stored as FLOAT[] arrays (vector representation).
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
