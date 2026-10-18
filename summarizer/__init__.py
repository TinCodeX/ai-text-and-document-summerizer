"""AI Document & Text Summarizer Package."""

from .chunker import chunk_text, summarize_large_document
from .config import ConfigurationError, Settings, get_llm_client, get_settings
from .ingestion import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    extract_text_from_file,
)
from .schema import SummaryFormat, SummaryResult
from .summarizer import EmptyInputError, SummarizationInferenceError, summarize_text

__all__ = [
    "Settings",
    "get_llm_client",
    "get_settings",
    "ConfigurationError",
    "SummaryFormat",
    "SummaryResult",
    "summarize_text",
    "EmptyInputError",
    "SummarizationInferenceError",
    "extract_text_from_file",
    "EmptyDocumentError",
    "UnsupportedFileTypeError",
    "chunk_text",
    "summarize_large_document",
]
