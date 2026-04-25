"""
File loader module.
Handles text extraction from PDF and DOCX files.
"""

import os
import re
from PyPDF2 import PdfReader
from docx import Document


def load_file(file_path: str) -> str:
    """
    Load a document file and return its cleaned text content.

    Supports .pdf and .docx formats.
    Raises ValueError for unsupported file types.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        raw_text = _extract_pdf(file_path)
    elif ext == ".docx":
        raw_text = _extract_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Use .pdf or .docx")

    cleaned = _clean_text(raw_text)

    if not cleaned.strip():
        raise ValueError(f"No text content found in: {file_path}")

    return cleaned


def _extract_pdf(file_path: str) -> str:
    """Extract text from all pages of a PDF file."""
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


def _extract_docx(file_path: str) -> str:
    """Extract text from all paragraphs of a DOCX file."""
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _clean_text(raw_text: str) -> str:
    """
    Clean extracted text:
    - Remove non-printable / control characters (keep newlines and tabs)
    - Collapse multiple whitespace into single spaces
    - Collapse multiple newlines into double newlines (paragraph breaks)
    - Strip leading/trailing whitespace
    """
    # Remove control characters except newline and tab
    text = re.sub(r'[^\S\n\t]+', ' ', raw_text)
    # Collapse multiple newlines into paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Remove leading/trailing whitespace
    return text.strip()
