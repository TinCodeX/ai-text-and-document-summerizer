"""
AI Document & Text Summarizer - Live Client Verification Test
=============================================================
Sends a minimal test prompt to the configured LLM client to verify:
1. Valid credentials
2. Network connectivity
3. Working inference response
"""

import sys
import time

# Windows console encoding safety
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel

try:
    from config import get_llm_client, ConfigurationError
except ImportError:
    # If run from outside the directory
    from summarizer.config import get_llm_client, ConfigurationError

console = Console(safe_box=True)

def main():
    console.print(Panel("[bold cyan]Step 1/2: Initializing LLM Client...[/bold cyan]", border_style="cyan"))

    try:
        client_wrapper = get_llm_client()
    except ConfigurationError as err:
        console.print(f"[bold red]Configuration Validation Failed:[/bold red] {err}")
        sys.exit(1)
    except Exception as err:
        console.print(f"[bold red]Client Initialization Failed:[/bold red] {err}")
        sys.exit(1)

    console.print(
        f"[bold green][OK][/bold green] Provider: [bold]{client_wrapper.provider.upper()}[/bold]\n"
        f"[bold green][OK][/bold green] Target Model: [bold]{client_wrapper.model}[/bold]\n"
        f"[bold green][OK][/bold green] Client Class: [dim]{client_wrapper.raw_client.__class__.__name__}[/dim]\n"
    )

    console.print(Panel("[bold cyan]Step 2/2: Executing Live Test Ping...[/bold cyan]", border_style="cyan"))
    
    test_prompt = "Say 'LLM Client Connected Successfully!' followed by a 1-sentence tip on summarization."
    
    start_time = time.perf_counter()
    try:
        with console.status("[bold yellow]Sending request to LLM API...[/bold yellow]"):
            response_text = client_wrapper.generate_text(
                prompt=test_prompt,
                system_instruction="You are a helpful AI connectivity assistant.",
            )
        elapsed = time.perf_counter() - start_time
    except Exception as err:
        console.print(
            Panel(
                f"[bold red]API Call Failed:[/bold red]\n{err}\n\n"
                "[dim]Check your internet connection, API key permissions, or model quota.[/dim]",
                title="Connection Error",
                border_style="red",
            )
        )
        sys.exit(1)

    console.print(
        Panel(
            f"[bold green]API Connection Verified![/bold green]\n\n"
            f"[bold]Model Response:[/bold]\n{response_text.strip()}\n\n"
            f"[dim]Latency: {elapsed:.2f}s | Provider: {client_wrapper.provider} | Model: {client_wrapper.model}[/dim]",
            title="Verification Result",
            border_style="green",
        )
    )

if __name__ == "__main__":
    main()
