"""
AI Document & Text Summarizer - Multi-Format Document Ingestion
===============================================================
Extracts clean, normalized textual content from diverse document formats
(.txt, .md, .pdf, .docx) with resilient decoding and validation guards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import docx
from pypdf import PdfReader


# ------------------------------------------------------------------------------
# 1. Custom Exceptions
# ------------------------------------------------------------------------------
class UnsupportedFileTypeError(ValueError):
    """Raised when the document extension is not supported by the ingestion pipeline."""


class EmptyDocumentError(ValueError):
    """Raised when the ingested file is physically empty or yields zero readable text."""


# ------------------------------------------------------------------------------
# 2. Format-Specific Extractors
# ------------------------------------------------------------------------------
def _extract_plain_text(path: Path) -> str:
    """Reads .txt and .md files with UTF-8 primary and latin-1 fallback encoding."""
    try:
        with open(path, mode="r", encoding="utf-8") as file_handle:
            return file_handle.read()
    except (UnicodeDecodeError, UnicodeError):
        with open(path, mode="r", encoding="latin-1", errors="replace") as file_handle:
            return file_handle.read()


def _extract_pdf_text(path: Path) -> str:
    """Extracts text page by page from PDF documents using pypdf."""
    reader = PdfReader(str(path))
    extracted_pages: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            extracted_pages.append(page_text.strip())

    return "\n\n".join(extracted_pages)


def _extract_docx_text(path: Path) -> str:
    """Extracts text across all paragraphs from DOCX documents using python-docx."""
    doc = docx.Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# ------------------------------------------------------------------------------
# 3. Public Ingestion API
# ------------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def extract_text_from_file(file_path: Union[str, Path]) -> str:
    """
    Extracts, normalizes, and validates text from supported file formats.

    Args:
        file_path: String path or Path object pointing to the target file.

    Returns:
        Extracted, stripped text string from the document.

    Raises:
        FileNotFoundError: If the specified file does not exist on disk.
        EmptyDocumentError: If the file size is 0 bytes or extracted text is empty.
        UnsupportedFileTypeError: If the file extension is not supported.
    """
    target = Path(file_path).resolve()

    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    if target.stat().st_size == 0:
        raise EmptyDocumentError(f"File '{target.name}' is physically empty (0 bytes).")

    ext = target.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file format '{ext}'. Supported formats are: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext in {".txt", ".md"}:
        raw_text = _extract_plain_text(target)
    elif ext == ".pdf":
        raw_text = _extract_pdf_text(target)
    elif ext == ".docx":
        raw_text = _extract_docx_text(target)
    else:
        raise UnsupportedFileTypeError(f"Unhandled file extension: '{ext}'")

    cleaned_text = raw_text.strip()
    if not cleaned_text:
        raise EmptyDocumentError(
            f"File '{target.name}' contains no readable or extractable text content."
        )

    return cleaned_text
