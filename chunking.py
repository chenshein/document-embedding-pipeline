"""
Text chunking module.

Strategy: Sentence-based splitting with overlap.

WHY SENTENCE-BASED?
We chose sentence-based splitting over the two alternatives:

1. Fixed-size splitting cuts text at arbitrary character positions, often breaking
   mid-sentence or mid-word. This destroys semantic meaning and produces embeddings
   that represent incomplete thoughts.

2. Paragraph-based splitting produces chunks of wildly inconsistent sizes — a single
   paragraph in a legal document might be 2000 words while a bullet point is 5 words.
   This inconsistency leads to poor and uneven embedding quality.

3. Sentence-based splitting (our choice) preserves complete semantic units. Each chunk
   contains full thoughts, the size is controllable via sentences_per_chunk, and the
   overlap ensures that context is not lost at chunk boundaries.
"""

import nltk

# Download sentence tokenizer data (only needed once)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


def chunk_text(
    text: str,
    sentences_per_chunk: int = 3,
    overlap: int = 1,
) -> list[str]:
    """
    Split text into overlapping sentence-based chunks.

    Args:
        text: The full document text.
        sentences_per_chunk: Number of sentences in each chunk.
        overlap: Number of sentences shared between consecutive chunks.

    Returns:
        A list of text chunks, each containing multiple sentences.
    """
    sentences = _split_sentences(text)

    if not sentences:
        return []

    # If the document is shorter than one chunk, return it as a single chunk
    if len(sentences) <= sentences_per_chunk:
        return [" ".join(sentences)]

    chunks = []
    step = sentences_per_chunk - overlap

    for start in range(0, len(sentences), step):
        end = start + sentences_per_chunk
        chunk_sentences = sentences[start:end]

        if not chunk_sentences:
            break

        # Merge tiny trailing chunks (< half the target size) into previous
        if len(chunk_sentences) < sentences_per_chunk // 2 and chunks:
            chunks[-1] = chunks[-1] + " " + " ".join(chunk_sentences)
            break

        chunks.append(" ".join(chunk_sentences))

        if end >= len(sentences):
            break

    return chunks


def _split_sentences(text: str) -> list[str]:
    """
    Split text into individual sentences using NLTK's sentence tokenizer.
    Filters out empty or whitespace-only results.
    """
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]
