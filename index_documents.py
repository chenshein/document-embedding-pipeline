"""
Main entry point for the document embedding pipeline.

Usage:
    python index_documents.py --init-db document.pdf          # first run
    python index_documents.py report.pdf notes.docx           # subsequent runs
"""

import argparse
import os
import sys

from file_loader import load_file
from chunking import chunk_text
from embeddings import generate_embeddings
from db import init_db, store_chunks
from config import CHUNK_SENTENCES, CHUNK_OVERLAP

SPLIT_STRATEGY = "sentence-based"


def process_document(file_path: str) -> int:
    """
    Run the full pipeline on a single document:
    1. Extract text from file
    2. Split into sentence-based chunks
    3. Generate embeddings via Gemini API
    4. Store in PostgreSQL

    Returns the number of chunks stored.
    """
    filename = os.path.basename(file_path)
    print(f"\n--- Processing: {filename} ---")

    # Step 1: Extract text
    print("  Extracting text...")
    text = load_file(file_path)
    print(f"  Extracted {len(text):,} characters")

    # Step 2: Chunk text
    print("  Chunking text...")
    chunks = chunk_text(text, sentences_per_chunk=CHUNK_SENTENCES, overlap=CHUNK_OVERLAP)
    print(f"  Created {len(chunks)} chunks (strategy: {SPLIT_STRATEGY})")

    # Step 3: Generate embeddings
    print("  Generating embeddings via Gemini API...")
    embeddings = generate_embeddings(chunks)
    print(f"  Generated {len(embeddings)} embeddings")

    # Step 4: Store in database
    print("  Storing in database...")
    count = store_chunks(filename, chunks, embeddings, SPLIT_STRATEGY)
    print(f"  Successfully stored {count} chunks")

    return count


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Index documents into PostgreSQL with vector embeddings."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to PDF or DOCX files to index",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database (create tables) before processing",
    )

    args = parser.parse_args()

    # Initialize database if requested
    if args.init_db:
        print("Initializing database...")
        init_db()

    # Process each file
    total_chunks = 0
    failures = 0

    for file_path in args.files:
        try:
            count = process_document(file_path)
            total_chunks += count
        except Exception as e:
            print(f"\n  ERROR processing {file_path}: {e}", file=sys.stderr)
            failures += 1

    # Summary
    succeeded = len(args.files) - failures
    print(f"\n=== Done: {succeeded}/{len(args.files)} files processed, "
          f"{total_chunks} total chunks stored ===")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
