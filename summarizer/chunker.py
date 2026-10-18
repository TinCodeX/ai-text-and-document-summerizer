"""
AI Document & Text Summarizer - Intelligent Chunking & Map-Reduce Pipeline
==========================================================================
Token/character-aware sliding window document chunking that honors natural
linguistic boundaries (paragraphs, sentences) and provides recursive Map-Reduce
summarization for large documents exceeding single context budgets.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    from .config import LLMClientWrapper, get_llm_client
    from .schema import SummaryFormat, SummaryResult
    from .summarizer import summarize_text
except (ImportError, ValueError):
    from config import LLMClientWrapper, get_llm_client
    from schema import SummaryFormat, SummaryResult
    from summarizer import summarize_text


# ------------------------------------------------------------------------------
# 1. Natural Boundary Sliding Window Chunker
# ------------------------------------------------------------------------------
def chunk_text(
    text: str,
    max_chunk_chars: int = 12000,
    overlap_chars: int = 500,
) -> list[str]:
    """
    Splits text into chunks fitting comfortably within character/token limits,
    preserving context via overlap and snapping to natural linguistic boundaries.

    Args:
        text: Raw document text to chunk.
        max_chunk_chars: Maximum character limit per chunk.
        overlap_chars: Overlap size preserved from preceding chunk.

    Returns:
        List of non-empty string chunks.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= max_chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start_idx = 0
    total_len = len(cleaned)

    # Boundary patterns in priority order: paragraph, line break, sentence, word
    boundary_patterns = [
        re.compile(r"\n\n+"),
        re.compile(r"\n"),
        re.compile(r"(?<=[.!?])\s+"),
        re.compile(r"\s+"),
    ]

    while start_idx < total_len:
        target_end = min(start_idx + max_chunk_chars, total_len)

        if target_end >= total_len:
            final_chunk = cleaned[start_idx:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        # Search backward from target_end for natural boundary
        search_window_start = max(start_idx + overlap_chars, target_end - 1500)
        search_slice = cleaned[search_window_start:target_end]
        split_point = -1

        for pattern in boundary_patterns:
            matches = list(pattern.finditer(search_slice))
            if matches:
                # Pick the latest boundary in the search window
                last_match = matches[-1]
                split_point = search_window_start + last_match.end()
                break

        if split_point == -1:
            split_point = target_end

        chunk_content = cleaned[start_idx:split_point].strip()
        if chunk_content:
            chunks.append(chunk_content)

        # Slide next window with overlap
        next_start = max(start_idx + 1, split_point - overlap_chars)
        # Advance past any leading whitespace in next chunk
        while next_start < total_len and cleaned[next_start].isspace():
            next_start += 1

        start_idx = next_start

    return chunks


# ------------------------------------------------------------------------------
# 2. Hierarchical Map-Reduce Summarization for Large Documents
# ------------------------------------------------------------------------------
def summarize_large_document(
    text: str,
    summary_format: SummaryFormat = SummaryFormat.BULLET_POINTS,
    chunk_size: int = 12000,
    client: Optional[LLMClientWrapper] = None,
) -> SummaryResult:
    """
    Summarizes documents of arbitrary scale:
    - If 1 chunk: Calls summarize_text directly.
    - If multi-chunk: Executes Map-Reduce pass across chunks before final synthesis.

    Args:
        text: Source document text.
        summary_format: Output format preset.
        chunk_size: Maximum characters per chunk.
        client: Optional pre-configured LLM client.

    Returns:
        Consolidated SummaryResult adhering to the Pydantic data contract.
    """
    chunks = chunk_text(text, max_chunk_chars=chunk_size)
    if not chunks:
        raise ValueError("Cannot summarize document yielding zero chunks.")

    active_client = client if client is not None else get_llm_client()

    # Base case: single chunk fits within context window
    if len(chunks) == 1:
        return summarize_text(
            text=chunks[0],
            summary_format=summary_format,
            client=active_client,
        )

    # --------------------------------------------------------------------------
    # MAP PHASE: Summarize individual chunks into high-density insights
    # --------------------------------------------------------------------------
    intermediate_summaries: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        map_prompt = (
            f"You are analyzing Section {idx} of {len(chunks)} of a large document.\n"
            "Extract the core facts, architectural decisions, metrics, and key statements "
            "from this section concisely in 3-5 dense bullet points:\n\n"
            f"\"\"\"\n{chunk}\n\"\"\""
        )
        section_summary = active_client.generate_text(
            prompt=map_prompt,
            system_instruction="You are a precise technical document analyzer.",
        )
        intermediate_summaries.append(
            f"--- SECTION {idx}/{len(chunks)} DIGEST ---\n{section_summary.strip()}"
        )

    # --------------------------------------------------------------------------
    # REDUCE PHASE: Aggregate intermediate digests into unified SummaryResult
    # --------------------------------------------------------------------------
    aggregated_context = "\n\n".join(intermediate_summaries)
    reduce_instructions = (
        f"This document was partitioned into {len(chunks)} sequential sections. "
        "Synthesize all sectional digests into a single cohesive global executive summary. "
        "Preserve overarching themes and consolidate action items."
    )

    final_result = summarize_text(
        text=aggregated_context,
        summary_format=summary_format,
        custom_instructions=reduce_instructions,
        client=active_client,
    )

    # Re-align raw original word count with entire source document
    final_result.word_count_original = len(text.strip().split())
    return final_result
