"""
AI Document & Text Summarizer - Automated Unit Tests
====================================================
Offline unit tests verifying schema validation, EmptyInputError handling,
and structured parsing logic using mock inference responses.
"""

import json
import sys
import unittest
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _PKG_DIR.parent
for p in (str(_PKG_DIR), str(_WORKSPACE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from summarizer.config import LLMClientWrapper
    from summarizer.schema import SummaryFormat, SummaryResult
    from summarizer.summarizer import EmptyInputError, summarize_text
except ImportError:
    from config import LLMClientWrapper
    from schema import SummaryFormat, SummaryResult
    from summarizer import EmptyInputError, summarize_text


class MockRawGeminiModel:
    def __init__(self, mock_response_text: str):
        self.mock_response_text = mock_response_text

    def generate_content(self, model: str, contents: str, config: any):
        class MockResponse:
            def __init__(self, text):
                self.text = text
        return MockResponse(self.mock_response_text)


class MockRawGeminiClient:
    def __init__(self, mock_response_text: str):
        self.models = MockRawGeminiModel(mock_response_text)


class TestSummarizerEngine(unittest.TestCase):

    def test_empty_input_raises_error(self):
        """Ensure whitespace or empty string raises EmptyInputError."""
        with self.assertRaises(EmptyInputError):
            summarize_text("   \n\t ")

    def test_schema_field_validation(self):
        """Verify field requirements and strip validator in SummaryResult."""
        data = {
            "title": "  Architecture Review 2026  ",
            "executive_summary": (
                "This is a comprehensive summary that satisfies minimum "
                "length requirements easily."
            ),
            "key_takeaways": [" Point A ", " Point B ", " "],
            "action_items": [" Action Item 1 "],
            "sentiment_or_tone": " Analytical ",
            "word_count_original": 150,
            "format_applied": "action_oriented",
        }
        model = SummaryResult(**data)
        self.assertEqual(model.title, "Architecture Review 2026")
        self.assertEqual(model.sentiment_or_tone, "Analytical")
        self.assertEqual(model.key_takeaways, ["Point A", "Point B"])
        self.assertEqual(model.action_items, ["Action Item 1"])

    def test_summarize_text_with_mock_client(self):
        """Verify structured parsing and metadata alignment with mock LLM client."""
        mock_payload = {
            "title": "Atlas Distributed Vector Pipeline",
            "executive_summary": (
                "Project Atlas redesigned the retrieval layer to reduce "
                "tail latency by 85%."
            ),
            "key_takeaways": [
                "Tail latency decreased from 850ms to 120ms.",
                "Dynamic vector quantization reduced storage by 22%.",
                "mTLS security signed off pending Monday review.",
            ],
            "action_items": [
                "Marcus to deprecate legacy RPC endpoints by Friday.",
                "DevOps to provision 2 read-replica clusters.",
            ],
            "sentiment_or_tone": "Technical & Decisive",
            "word_count_original": 0,
            "format_applied": "bullet_points",
        }

        mock_client = LLMClientWrapper(
            client=MockRawGeminiClient(json.dumps(mock_payload)),
            provider="gemini",
            model="gemini-2.0-flash",
            temperature=0.2,
        )

        sample_text = "This is a sample document for testing the summarizer engine."
        result = summarize_text(
            text=sample_text,
            summary_format=SummaryFormat.ACTION_ORIENTED,
            client=mock_client,
        )

        self.assertIsInstance(result, SummaryResult)
        self.assertEqual(result.title, "Atlas Distributed Vector Pipeline")
        self.assertEqual(result.word_count_original, 10)  # Counted accurately
        self.assertEqual(result.format_applied, "action_oriented")
        self.assertEqual(len(result.key_takeaways), 3)
        self.assertEqual(len(result.action_items), 2)


if __name__ == "__main__":
    unittest.main()
