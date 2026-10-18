"""
AI Document & Text Summarizer - Structured Data Contracts
=========================================================
Pydantic schemas and enumeration contracts enforcing strict typing, field
validation, and structured LLM responses.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class SummaryFormat(str, Enum):
    """Supported summarization presentation styles."""

    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"
    DETAILED = "detailed"
    ACTION_ORIENTED = "action_oriented"

    def get_prompt_instruction(self) -> str:
        """Returns format-specific guiding instructions for prompt engineering."""
        instructions = {
            SummaryFormat.BULLET_POINTS: (
                "Format focus: Prioritize high-impact, scannable bullet points in key_takeaways."
            ),
            SummaryFormat.EXECUTIVE: (
                "Format focus: Provide a comprehensive, high-level strategic overview in "
                "executive_summary suitable for C-level leadership."
            ),
            SummaryFormat.DETAILED: (
                "Format focus: Provide granular technical depth, key nuances, and thorough takeaways."
            ),
            SummaryFormat.ACTION_ORIENTED: (
                "Format focus: Extract and emphasize all direct tasks, deadlines, owners, "
                "and recommendations inside action_items."
            ),
        }
        return instructions[self]


class SummaryResult(BaseModel):
    """
    Validated structured summary payload returned by the inference engine.
    Guarantees strict schema adherence across Gemini and OpenAI providers.
    """

    title: str = Field(
        ...,
        description="A concise, punchy title characterizing the core theme of the text.",
        min_length=3,
        max_length=120,
    )

    executive_summary: str = Field(
        ...,
        description="A cohesive 2 to 3 sentence synthesis capturing the essence of the source document.",
        min_length=20,
    )

    key_takeaways: List[str] = Field(
        default_factory=list,
        description="A curated list of 3 to 5 core insights or primary arguments.",
    )

    action_items: List[str] = Field(
        default_factory=list,
        description="Actionable follow-ups, directives, decisions, or next steps extracted from text.",
    )

    sentiment_or_tone: str = Field(
        ...,
        description="Tone and communication style descriptor (e.g., 'Analytical', 'Urgent', 'Informative').",
        min_length=3,
        max_length=50,
    )

    word_count_original: int = Field(
        default=0,
        ge=0,
        description="Raw word count of the source input text processed by the engine.",
    )

    format_applied: str = Field(
        default=SummaryFormat.BULLET_POINTS.value,
        description="The summarization style preset applied during generation.",
    )

    @field_validator("title", "executive_summary", "sentiment_or_tone", mode="before")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Trims extraneous leading/trailing whitespace from string fields."""
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("key_takeaways", "action_items", mode="before")
    @classmethod
    def clean_string_list(cls, values: list) -> list:
        """Filters out empty or whitespace-only items in string collections."""
        if isinstance(values, list):
            return [str(item).strip() for item in values if str(item).strip()]
        return []
