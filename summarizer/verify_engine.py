"""
AI Document & Text Summarizer - Core Inference Engine Verification
==================================================================
Tests structured Pydantic summarization, validates type contracts,
and displays formatted outputs using Rich tables and panels.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root and package directory to sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _CURRENT_DIR.parent
for _path in (str(_CURRENT_DIR), str(_WORKSPACE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Windows console encoding safety
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from summarizer.config import ConfigurationError
    from summarizer.schema import SummaryFormat, SummaryResult
    from summarizer.summarizer import EmptyInputError, summarize_text
except ImportError:
    from config import ConfigurationError
    from schema import SummaryFormat, SummaryResult
    from summarizer import EmptyInputError, summarize_text

console = Console(safe_box=True)

# ------------------------------------------------------------------------------
# Sample Text (~250 words): Engineering Architecture & Sprint Briefing
# ------------------------------------------------------------------------------
SAMPLE_CORPUS = """
Sprint 42 Architecture Review & Project Atlas Deployment Briefing

During today's sync, the platform engineering team concluded the architectural 
redesign for Project Atlas, our distributed vector processing pipeline. Over the 
past quarter, query latency spiked by 38% due to unindexed semantic embeddings 
under peak ingestion loads. 

Lead Infrastructure Architect Sarah Jenkins presented the benchmarking data. 
Migrating the retrieval layer to an asynchronous batching worker model reduced 
p99 tail latency from 850ms down to 120ms during simulated 50,000 req/sec stress tests. 
Additionally, storage overhead was decreased by 22% through dynamic vector quantization.

Key decisions and next steps:
1. Marcus will finish deprecating the legacy sync RPC endpoints by Friday, October 24.
2. The DevOps team must provision two additional read-replica database clusters in us-east-2.
3. Elena will update the integration testing harness to validate schema contracts against 
   the new gRPC payload definitions before next Tuesday.
4. Security auditing signed off on mTLS encryption across all internal pod communication, 
   pending a final penetration review on Monday morning.

Overall, the migration trajectory remains on schedule for full production cutover by 
the end of next month, provided all staging validation checks pass without regression.
"""


def verify_empty_input_handling() -> None:
    """Validates that empty inputs properly raise EmptyInputError."""
    console.print("[dim]Testing empty input validation...[/dim]")
    try:
        summarize_text("   \n\t  ")
        raise AssertionError("EmptyInputError was NOT raised for whitespace string!")
    except EmptyInputError:
        console.print("[bold green][OK][/bold green] Empty input error correctly trapped.")


def display_summary_results(summary: SummaryResult, elapsed_time: float) -> None:
    """Pretty prints the validated SummaryResult in an executive dashboard format."""
    # Metadata Overview Table
    meta_table = Table(title="Execution & Contract Metadata", show_header=True)
    meta_table.add_column("Attribute", style="bold cyan")
    meta_table.add_column("Value", style="green")

    meta_table.add_row("Extracted Title", summary.title)
    meta_table.add_row("Detected Tone", summary.sentiment_or_tone)
    meta_table.add_row("Format Applied", summary.format_applied)
    meta_table.add_row("Original Word Count", str(summary.word_count_original))
    meta_table.add_row("Inference Latency", f"{elapsed_time:.2f} seconds")
    console.print(meta_table)

    # Executive Summary Panel
    console.print(
        Panel(
            summary.executive_summary,
            title="[bold blue]Executive Summary[/bold blue]",
            border_style="blue",
        )
    )

    # Key Takeaways Table
    takeaways_table = Table(title="Key Takeaways", show_header=False)
    takeaways_table.add_column("Index", style="bold yellow", width=4)
    takeaways_table.add_column("Takeaway", style="white")
    for i, point in enumerate(summary.key_takeaways, 1):
        takeaways_table.add_row(f"{i}.", point)
    console.print(takeaways_table)

    # Action Items Table
    if summary.action_items:
        actions_table = Table(title="Action Items & Directives", show_header=False)
        actions_table.add_column("Index", style="bold magenta", width=4)
        actions_table.add_column("Action Item", style="white")
        for i, item in enumerate(summary.action_items, 1):
            actions_table.add_row(f"[ ]", item)
        console.print(actions_table)


def run_verification() -> None:
    """Main verification routine."""
    console.print(
        Panel.fit(
            "[bold cyan]AI Document & Text Summarizer[/bold cyan]\n"
            "[dim]Task 2: Structured Data Contracts & Inference Engine Verification[/dim]",
            border_style="cyan",
        )
    )

    # 1. Test Input Guard
    verify_empty_input_handling()

    # 2. Test Inference with Structured Output
    use_mock = "--mock" in sys.argv or "--demo" in sys.argv
    if use_mock:
        console.print("[bold yellow]Running in demonstration mode (--mock)...[/bold yellow]")
        import json
        from summarizer.config import LLMClientWrapper

        class _MockModel:
            def generate_content(self, model, contents, config):
                class _Resp:
                    text = json.dumps({
                        "title": "Project Atlas Distributed Vector Pipeline Architecture Review",
                        "executive_summary": (
                            "The platform engineering team finalized the Project Atlas architectural "
                            "overhaul, migrating the retrieval layer to an asynchronous batching worker "
                            "model. This cut p99 tail latency from 850ms down to 120ms during peak load tests "
                            "and reduced storage overhead by 22% via dynamic quantization."
                        ),
                        "key_takeaways": [
                            "Tail query latency dropped from 850ms to 120ms at 50,000 req/sec load.",
                            "Dynamic vector quantization reduced overall storage overhead by 22%.",
                            "Legacy sync RPC endpoints scheduled for deprecation by October 24.",
                            "mTLS security encryption approved across internal pod communication.",
                        ],
                        "action_items": [
                            "Marcus: Deprecate legacy sync RPC endpoints by Friday, Oct 24.",
                            "DevOps Team: Provision 2 additional read-replica database clusters in us-east-2.",
                            "Elena: Update integration test harness for gRPC payloads before next Tuesday.",
                            "Security Team: Complete final Monday penetration testing review for mTLS.",
                        ],
                        "sentiment_or_tone": "Technical & Action-Oriented",
                        "word_count_original": 178,
                        "format_applied": "action_oriented",
                    })
                return _Resp()

        class _MockClient:
            models = _MockModel()

        mock_wrapper = LLMClientWrapper(
            client=_MockClient(),
            provider="gemini",
            model="gemini-2.0-flash",
            temperature=0.2,
        )
        start_time = time.perf_counter()
        summary = summarize_text(
            text=SAMPLE_CORPUS,
            summary_format=SummaryFormat.ACTION_ORIENTED,
            custom_instructions="Highlight engineering owners and specific deadlines.",
            client=mock_wrapper,
        )
        elapsed = time.perf_counter() - start_time
    else:
        console.print("\n[bold cyan]Executing live summarize_text() with structured schema enforcement...[/bold cyan]")
        start_time = time.perf_counter()

        try:
            summary = summarize_text(
                text=SAMPLE_CORPUS,
                summary_format=SummaryFormat.ACTION_ORIENTED,
                custom_instructions="Highlight engineering owners and specific deadlines.",
            )
            elapsed = time.perf_counter() - start_time
        except ConfigurationError as err:
            console.print(
                Panel(
                    f"[bold red]Configuration Missing:[/bold red]\n{err}\n\n"
                    "[dim]Tip: You can run [bold cyan]python summarizer/verify_engine.py --mock[/bold cyan] "
                    "to test the full pipeline and Rich display offline, or copy .env.example to .env to test live API calls.[/dim]",
                    title="Credential Check",
                    border_style="yellow",
                )
            )
            sys.exit(1)
        except Exception as err:
            console.print(
                Panel(
                    f"[bold red]Inference Engine Error:[/bold red]\n{err}",
                title="Execution Failed",
                border_style="red",
            )
        )
        sys.exit(1)

    # 3. Assert Pydantic Type Contracts
    assert isinstance(summary, SummaryResult), "Output must be an instance of SummaryResult"
    assert len(summary.title) >= 3, "Title too short"
    assert len(summary.executive_summary) >= 20, "Executive summary too short"
    assert isinstance(summary.key_takeaways, list), "key_takeaways must be a list"
    assert isinstance(summary.action_items, list), "action_items must be a list"
    assert summary.word_count_original > 0, "Word count must be greater than 0"
    assert summary.format_applied == SummaryFormat.ACTION_ORIENTED.value

    # 4. Render Rich Dashboard
    console.print("\n[bold green][OK] Pydantic Contract Assertions Passed Successfully![/bold green]\n")
    display_summary_results(summary, elapsed)


if __name__ == "__main__":
    run_verification()
