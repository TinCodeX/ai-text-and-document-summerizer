"""
AI Document & Text Summarizer - Production CLI Application
==========================================================
Interactive & non-interactive terminal interface for multi-format document
ingestion, structured LLM summarization, and automated file exports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

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
from rich.prompt import Prompt
from rich.table import Table

try:
    from summarizer.chunker import chunk_text, summarize_large_document
    from summarizer.config import ConfigurationError, LLMClientWrapper, get_llm_client
    from summarizer.exporters import export_summary
    from summarizer.ingestion import (
        EmptyDocumentError,
        UnsupportedFileTypeError,
        extract_text_from_file,
    )
    from summarizer.schema import SummaryFormat, SummaryResult
    from summarizer.summarizer import EmptyInputError, summarize_text
except ImportError:
    from chunker import chunk_text, summarize_large_document
    from config import ConfigurationError, LLMClientWrapper, get_llm_client
    from exporters import export_summary
    from ingestion import (
        EmptyDocumentError,
        UnsupportedFileTypeError,
        extract_text_from_file,
    )
    from schema import SummaryFormat, SummaryResult
    from summarizer import EmptyInputError, summarize_text

console = Console(safe_box=True)


# ------------------------------------------------------------------------------
# 1. Synthetic Client for Mock / Demonstration Mode
# ------------------------------------------------------------------------------
def _create_mock_client(text_preview: str, style: str) -> LLMClientWrapper:
    """Creates a local mock LLM client returning realistic structured summaries."""
    class _MockModel:
        def generate_content(self, model, contents, config):
            class _Resp:
                text = json.dumps({
                    "title": "Document Intelligence & Strategic Briefing",
                    "executive_summary": (
                        "This document highlights core operational paradigms, technical architecture "
                        "milestones, and execution risks. The findings recommend prioritizing immediate "
                        "resource allocation to resolve throughput constraints and accelerate timelines."
                    ),
                    "key_takeaways": [
                        "Identified latency and scaling bottlenecks under peak operational loads.",
                        "Architectural decoupling improved system resilience and reduced blast radius.",
                        "Cross-functional teams aligned on key milestones ahead of production launch.",
                    ],
                    "action_items": [
                        "Conduct end-to-end integration stress tests by end of week.",
                        "Finalize security audits and mTLS configurations across internal services.",
                        "Schedule deployment readiness review with stakeholders.",
                    ],
                    "sentiment_or_tone": "Strategic & Analytical",
                    "word_count_original": len(text_preview.split()),
                    "format_applied": style,
                })
            return _Resp()

    class _MockRawClient:
        models = _MockModel()

    return LLMClientWrapper(
        client=_MockRawClient(),
        provider="gemini",
        model="mock-fast-model",
        temperature=0.2,
    )


# ------------------------------------------------------------------------------
# 2. Terminal Display & Presentation
# ------------------------------------------------------------------------------
def display_summary(summary: SummaryResult, elapsed: float, source_desc: str) -> None:
    """Renders the executive summary dashboard using Rich components."""
    # Metadata Overview
    meta_table = Table(show_header=False, box=None)
    meta_table.add_column("Key", style="bold cyan", width=22)
    meta_table.add_column("Val", style="green")

    meta_table.add_row("Document Title:", summary.title)
    meta_table.add_row("Source:", source_desc)
    meta_table.add_row("Format Applied:", summary.format_applied)
    meta_table.add_row("Analyzed Tone:", summary.sentiment_or_tone)
    meta_table.add_row("Source Word Count:", f"{summary.word_count_original:,} words")
    meta_table.add_row("Processing Latency:", f"{elapsed:.2f} seconds")

    console.print(
        Panel(
            meta_table,
            title="[bold cyan]Document Intelligence Synthesis[/bold cyan]",
            border_style="cyan",
        )
    )

    # Executive Summary
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
    for i, pt in enumerate(summary.key_takeaways, 1):
        takeaways_table.add_row(f"{i}.", pt)
    console.print(takeaways_table)

    # Action Items Table
    if summary.action_items:
        actions_table = Table(title="Action Items & Directives", show_header=False)
        actions_table.add_column("Box", style="bold magenta", width=5)
        actions_table.add_column("Directive", style="white")
        for item in summary.action_items:
            actions_table.add_row("[ ]", item)
        console.print(actions_table)


def display_export_locations(exported_files: dict[str, Path]) -> None:
    """Prints a table of generated artifacts."""
    if not exported_files:
        return

    table = Table(title="Export Deliverables", show_header=True)
    table.add_column("Format", style="bold yellow", width=10)
    table.add_column("Saved File Path", style="dim green")

    for fmt, p in exported_files.items():
        table.add_row(fmt.upper(), str(p))

    console.print(table)


# ------------------------------------------------------------------------------
# 3. Interactive Wizard Flow
# ------------------------------------------------------------------------------
def run_interactive_wizard() -> tuple[str, str, SummaryFormat, Optional[str], list[str]]:
    """Guides the user through an interactive prompt flow."""
    console.print(
        Panel.fit(
            "[bold cyan]AI Document & Text Summarizer[/bold cyan]\n"
            "[dim]Interactive Briefing Wizard[/dim]",
            border_style="cyan",
        )
    )

    input_mode = Prompt.ask(
        "Select input source",
        choices=["file", "text"],
        default="file",
    )

    source_desc = ""
    raw_text = ""

    if input_mode == "file":
        while True:
            file_input = Prompt.ask("Enter document path (.txt, .md, .pdf, .docx)")
            candidate = Path(file_input.strip('"\'')).resolve()
            if candidate.exists() and candidate.is_file():
                try:
                    with console.status("[cyan]Reading document...[/cyan]"):
                        raw_text = extract_text_from_file(candidate)
                    source_desc = f"File: {candidate.name}"
                    break
                except Exception as err:
                    console.print(f"[bold red]Extraction Error:[/bold red] {err}")
            else:
                console.print(f"[bold red]Error:[/bold red] File not found at '{candidate}'. Try again.")
    else:
        console.print("[dim]Paste your text below (Press Enter, then Ctrl+Z / Ctrl+D or type EOF on a new line):[/dim]")
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
        except EOFError:
            pass
        raw_text = "\n".join(lines).strip()
        source_desc = "Pasted Text Input"

    style_choice = Prompt.ask(
        "Select summary style",
        choices=["bullet_points", "executive", "detailed", "action_oriented"],
        default="bullet_points",
    )
    summary_format = SummaryFormat(style_choice)

    instructions = Prompt.ask(
        "Custom focus instructions (optional)",
        default="",
    ).strip() or None

    export_choices = Prompt.ask(
        "Export formats (comma-separated: md,json,txt or 'none')",
        default="md,json",
    ).strip().lower()

    if export_choices in {"none", "no", "false"}:
        formats: list[str] = []
    else:
        formats = [f.strip() for f in export_choices.split(",") if f.strip()]

    return raw_text, source_desc, summary_format, instructions, formats


# ------------------------------------------------------------------------------
# 4. CLI Argument Parsing & Orchestration
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="summarizer",
        description="AI Document & Text Summarizer: Production CLI for multi-format documents.",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to source document (.txt, .md, .pdf, .docx)",
    )
    parser.add_argument(
        "-t", "--text",
        type=str,
        help="Direct plain text content to summarize",
    )
    parser.add_argument(
        "-s", "--format",
        choices=["bullet_points", "executive", "detailed", "action_oriented"],
        default="bullet_points",
        help="Summarization style preset (default: bullet_points)",
    )
    parser.add_argument(
        "-i", "--instructions",
        type=str,
        help="Optional custom instructions to guide LLM focus",
    )
    parser.add_argument(
        "-e", "--export",
        type=str,
        default="md,json",
        help="Comma-separated export formats: md, json, txt, or all (default: md,json)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing files and only display output in console",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal wizard",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic responses for offline testing without consuming API quota",
    )
    return parser


def main() -> None:
    """Main CLI execution flow."""
    parser = build_parser()
    args = parser.parse_args()

    # Determine if interactive mode is triggered
    has_input = bool(args.file or args.text)
    if args.interactive or not has_input:
        raw_text, source_desc, summary_format, custom_instructions, export_formats = (
            run_interactive_wizard()
        )
        no_export = len(export_formats) == 0
    else:
        summary_format = SummaryFormat(args.format)
        custom_instructions = args.instructions
        no_export = args.no_export
        export_formats = [f.strip() for f in args.export.split(",") if f.strip()]

        if args.file:
            target_path = Path(args.file.strip('"\'')).resolve()
            source_desc = f"File: {target_path.name}"
            with console.status("[bold cyan]Extracting text from file...[/bold cyan]"):
                try:
                    raw_text = extract_text_from_file(target_path)
                except Exception as err:
                    console.print(f"[bold red]File Ingestion Failed:[/bold red] {err}")
                    sys.exit(1)
        else:
            raw_text = (args.text or "").strip()
            source_desc = "Inline CLI Text"

    if not raw_text or not raw_text.strip():
        console.print("[bold red]Error:[/bold red] No text content provided to summarize.")
        sys.exit(1)

    # Resolve Client Instance
    if args.mock:
        client = _create_mock_client(raw_text, summary_format.value)
    else:
        try:
            client = get_llm_client()
        except ConfigurationError as err:
            console.print(
                Panel(
                    f"[bold red]Configuration Missing:[/bold red]\n{err}\n\n"
                    "[dim]Tip: You can use the [bold cyan]--mock[/bold cyan] flag to test the full CLI "
                    "offline without an API key, or populate summarizer/.env with your key.[/dim]",
                    title="Setup Required",
                    border_style="yellow",
                )
            )
            sys.exit(1)

    # Execute Summarization (Direct or Map-Reduce)
    char_count = len(raw_text)
    start_time = time.perf_counter()

    try:
        if char_count > 12000:
            with console.status(
                f"[bold yellow]Large document ({char_count:,} chars) - running Map-Reduce pipeline...[/bold yellow]"
            ):
                summary = summarize_large_document(
                    text=raw_text,
                    summary_format=summary_format,
                    chunk_size=12000,
                    client=client,
                )
        else:
            with console.status("[bold yellow]Synthesizing structured summary with LLM...[/bold yellow]"):
                summary = summarize_text(
                    text=raw_text,
                    summary_format=summary_format,
                    custom_instructions=custom_instructions,
                    client=client,
                )
        elapsed = time.perf_counter() - start_time
    except Exception as err:
        console.print(Panel(f"[bold red]Summarization Engine Error:[/bold red]\n{err}", border_style="red"))
        sys.exit(1)

    # Display Results in Terminal
    console.print()
    display_summary(summary, elapsed, source_desc)

    # Multi-Format Export
    if not no_export:
        source_prefix = Path(args.file).stem if args.file else "text"
        with console.status("[bold cyan]Exporting summary files...[/bold cyan]"):
            written_files = export_summary(
                result=summary,
                output_dir="output",
                file_prefix=source_prefix,
                formats=export_formats,
            )
        console.print()
        display_export_locations(written_files)


if __name__ == "__main__":
    main()
