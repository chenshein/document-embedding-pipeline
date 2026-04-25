"""
Test script to verify that embeddings and similarity search work correctly.

This script:
1. Embeds a search query using the same Gemini model
2. Compares it against all stored chunks using cosine distance (pgvector)
3. Returns the most semantically similar chunks — proving the pipeline works
"""

from config import POSTGRES_URL
from embeddings import generate_embeddings
from sqlalchemy import create_engine, text

engine = create_engine(POSTGRES_URL)


def search(query: str, top_k: int = 3):
    """
    Semantic search: embed the query and find the closest chunks in the database
    using pgvector's cosine distance operator (<=>).
    """
    # Generate embedding for the search query
    query_embedding = generate_embeddings([query])[0]

    # Convert to pgvector format: [0.1, 0.2, ...]
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with engine.connect() as conn:
        results = conn.execute(
            text("""
                SELECT id, filename, left(chunk_text, 200) AS preview,
                       1 - (embedding <=> :vec) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> :vec
                LIMIT :k
            """),
            {"vec": vec_str, "k": top_k},
        ).fetchall()

    return results


if __name__ == "__main__":
    query = input("Enter a search query: ")
    print(f"\nSearching for: '{query}'\n")

    results = search(query)

    if not results:
        print("No chunks found in database. Run index_documents.py first.")
    else:
        for r in results:
            print(f"--- Chunk {r[0]} (similarity: {r[3]:.4f}) ---")
            print(f"  File: {r[1]}")
            print(f"  Preview: {r[2]}...")
            print()
