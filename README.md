# Document Embedding Pipeline

A Python module that extracts text from PDF/DOCX documents, splits it into semantic chunks, generates vector embeddings using Google Gemini, and stores everything in PostgreSQL with pgvector for semantic search.

## Architecture

```
PDF/DOCX file
    → file_loader.py    (extract & clean text)
    → chunking.py       (split into sentence-based chunks)
    → embeddings.py     (generate vectors via Gemini API)
    → db.py             (store in PostgreSQL with pgvector)
```

`index_documents.py` is the CLI entry point that orchestrates the pipeline.

## Chunking Strategy: Sentence-Based Splitting

We use **sentence-based splitting** with a sliding window of 3 sentences and 1-sentence overlap.

### Why not fixed-size splitting?
Fixed-size splitting cuts 
text at arbitrary character positions, often breaking mid-sentence or mid-word. This destroys semantic meaning and produces embeddings that represent incomplete thoughts — leading to poor search results.

### Why not paragraph-based splitting?
Paragraph sizes vary wildly — a legal paragraph can be 2000 words while a bullet point is 5 words. This inconsistency produces embeddings of uneven quality, making similarity comparisons unreliable.

### Why sentence-based?
Sentence-based splitting preserves **complete semantic units**. Each chunk contains full thoughts, the chunk size is consistent and controllable via `sentences_per_chunk`, and the overlap ensures no context is lost at chunk boundaries. This produces high-quality, consistent embeddings ideal for semantic search.

## Project Structure

```
part2/
├── index_documents.py   # CLI entry point — orchestrates the pipeline
├── config.py            # Loads .env, exposes settings as constants
├── file_loader.py       # PDF/DOCX text extraction and cleaning
├── chunking.py          # Sentence-based text chunking with overlap
├── embeddings.py        # Gemini API embedding generation
├── db.py                # PostgreSQL + pgvector storage (SQLAlchemy ORM)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
└── README.md
```

## Setup

### 1. Prerequisites
- Python 3.10+
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- A Google Gemini API key

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:
```
GEMINI_API_KEY=your-actual-gemini-api-key
POSTGRES_URL=postgresql://user:password@localhost:5432/embeddings_db
```

### 4. Create the database

```sql
CREATE DATABASE embeddings_db;
```

## Usage

### First run (initialize tables and pgvector extension):
```bash
python index_documents.py --init-db document.pdf
```

### Index one or more documents:
```bash
python index_documents.py report.pdf notes.docx
```

### View stored chunks in PostgreSQL:
```sql
-- See all chunks with metadata
SELECT id, filename, split_strategy, left(chunk_text, 100) AS preview, created_at
FROM document_chunks;

-- See full text of a specific chunk
SELECT chunk_text FROM document_chunks WHERE id = 1;

-- Count chunks per document
SELECT filename, COUNT(*) AS chunk_count FROM document_chunks GROUP BY filename;
```

## Database Schema

| Column         | Type          | Description                     |
|----------------|---------------|---------------------------------|
| id             | SERIAL (PK)   | Auto-incrementing primary key  |
| chunk_text     | TEXT          | The text content of the chunk   |
| embedding      | VECTOR(3072)  | 3072-dimensional vector embedding (pgvector) |
| filename       | TEXT          | Source document name            |
| split_strategy | TEXT          | Chunking method used            |
| created_at     | TIMESTAMP     | Row creation time (auto-filled) |
