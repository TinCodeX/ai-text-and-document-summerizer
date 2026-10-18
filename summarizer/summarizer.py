"""
AI Document & Text Summarizer - Core Inference Engine
=====================================================
Executes structured document summarization with Pydantic contract enforcement
across Google Gemini and OpenAI model families.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from google.genai import types as genai_types
from pydantic import ValidationError

try:
    from .config import LLMClientWrapper, get_llm_client
    from .schema import SummaryFormat, SummaryResult
except (ImportError, ValueError):
    from config import LLMClientWrapper, get_llm_client
    from schema import SummaryFormat, SummaryResult


# ------------------------------------------------------------------------------
# 1. Custom Exceptions
# ------------------------------------------------------------------------------
class EmptyInputError(ValueError):
    """Raised when the text provided for summarization is empty or whitespace only."""


class SummarizationInferenceError(RuntimeError):
    """Raised when model inference or JSON schema parsing fails."""


# ------------------------------------------------------------------------------
# 2. Text Normalization & Helper Utilities
# ------------------------------------------------------------------------------
def _clean_json_markdown(raw_text: str) -> str:
    """
    Strips Markdown code block delimiters (e.g. ```json ... ```) from LLM output.
    """
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def _build_prompt_payload(
    text: str,
    summary_format: SummaryFormat,
    custom_instructions: Optional[str] = None,
) -> tuple[str, str]:
    """
    Constructs the system prompt and user query payload for summarization.
    """
    system_instruction = (
        "You are an elite AI document analysis and executive summarization system. "
        "Your task is to accurately extract key intelligence, synthesize themes, "
        "and produce a comprehensive structured summary strictly conforming to the JSON schema. "
        "Never hallucinate facts not present in the source text."
    )

    prompt_parts = [
        f"SOURCE TEXT TO SUMMARIZE:\n\"\"\"\n{text.strip()}\n\"\"\"",
        f"\nSTYLE & FORMATTING DIRECTIVE:\n{summary_format.get_prompt_instruction()}",
    ]

    if custom_instructions and custom_instructions.strip():
        prompt_parts.append(
            f"\nADDITIONAL USER INSTRUCTIONS:\n{custom_instructions.strip()}"
        )

    prompt_parts.append(
        "\nIMPORTANT: Respond ONLY with a valid JSON object matching the required schema."
    )

    user_prompt = "\n".join(prompt_parts)
    return system_instruction, user_prompt


# ------------------------------------------------------------------------------
# 3. Provider-Specific Execution Handlers
# ------------------------------------------------------------------------------
def _execute_gemini_summary(
    client_wrapper: LLMClientWrapper,
    system_instruction: str,
    user_prompt: str,
) -> SummaryResult:
    """Invokes Google Gemini with native response_schema enforcement."""
    config = genai_types.GenerateContentConfig(
        temperature=client_wrapper.temperature,
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=SummaryResult,
    )

    response = client_wrapper.raw_client.models.generate_content(
        model=client_wrapper.model,
        contents=user_prompt,
        config=config,
    )

    raw_output = response.text or ""
    cleaned_json = _clean_json_markdown(raw_output)

    try:
        return SummaryResult.model_validate_json(cleaned_json)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise SummarizationInferenceError(
            f"Failed to validate Gemini structured output against schema: {exc}\nRaw: {raw_output}"
        ) from exc


def _execute_openai_summary(
    client_wrapper: LLMClientWrapper,
    system_instruction: str,
    user_prompt: str,
) -> SummaryResult:
    """Invokes OpenAI with structured response_format parsing."""
    try:
        completion = client_wrapper.raw_client.beta.chat.completions.parse(
            model=client_wrapper.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            response_format=SummaryResult,
            temperature=client_wrapper.temperature,
        )
        parsed_result = completion.choices[0].message.parsed
        if parsed_result is not None:
            return parsed_result
    except Exception:
        # Fallback to standard chat completion with manual schema parsing
        pass

    # Fallback completion call
    completion = client_wrapper.raw_client.chat.completions.create(
        model=client_wrapper.model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=client_wrapper.temperature,
    )
    raw_output = completion.choices[0].message.content or ""
    cleaned_json = _clean_json_markdown(raw_output)

    try:
        return SummaryResult.model_validate_json(cleaned_json)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise SummarizationInferenceError(
            f"Failed to parse OpenAI response into SummaryResult schema: {exc}\nRaw: {raw_output}"
        ) from exc


# ------------------------------------------------------------------------------
# 4. Public API Entrypoint
# ------------------------------------------------------------------------------
def summarize_text(
    text: str,
    summary_format: SummaryFormat = SummaryFormat.BULLET_POINTS,
    custom_instructions: Optional[str] = None,
    client: Optional[LLMClientWrapper] = None,
) -> SummaryResult:
    """
    Summarizes arbitrary text into a validated SummaryResult Pydantic contract.

    Args:
        text: Source text to summarize. Must contain non-whitespace characters.
        summary_format: Style preset (BULLET_POINTS, EXECUTIVE, DETAILED, ACTION_ORIENTED).
        custom_instructions: Optional bespoke prompt directives from the caller.
        client: Optional pre-configured LLMClientWrapper. If None, initialized from config.

    Returns:
        SummaryResult containing validated title, executive summary, takeaways,
        action items, sentiment, word count, and applied format.

    Raises:
        EmptyInputError: If text is empty or contains only whitespace.
        SummarizationInferenceError: If model call or schema validation fails.
    """
    if not text or not text.strip():
        raise EmptyInputError(
            "Cannot summarize empty text. Please provide valid text content."
        )

    # Compute raw word count accurately from source text
    raw_word_count = len(text.strip().split())

    active_client = client if client is not None else get_llm_client()
    system_instruction, user_prompt = _build_prompt_payload(
        text=text,
        summary_format=summary_format,
        custom_instructions=custom_instructions,
    )

    if active_client.provider == "gemini":
        result = _execute_gemini_summary(
            client_wrapper=active_client,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
        )
    elif active_client.provider == "openai":
        result = _execute_openai_summary(
            client_wrapper=active_client,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
        )
    else:
        raise SummarizationInferenceError(
            f"Unsupported LLM provider: {active_client.provider}"
        )

    # Guarantee accurate metadata alignment
    result.word_count_original = raw_word_count
    result.format_applied = summary_format.value
    return result
