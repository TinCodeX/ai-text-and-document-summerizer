"""
AI Document & Text Summarizer - Configuration & LLM Client Loader
=================================================================
Safely parses environment variables, validates provider credentials, and
provides a modular factory for authenticated inference clients (Gemini / OpenAI).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

# Windows console encoding safety (prevents charmap/cp1252 crashes)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel

console = Console(safe_box=True)

# ------------------------------------------------------------------------------
# 1. Environment Discovery & Loading
# ------------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_ENV_PATH = find_dotenv(usecwd=True) or str(_CURRENT_DIR / ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=False)


# ------------------------------------------------------------------------------
# 2. Custom Exceptions
# ------------------------------------------------------------------------------
class ConfigurationError(Exception):
    """Raised when application configuration or API keys are invalid/missing."""


# ------------------------------------------------------------------------------
# 3. Settings Schema Definition (Pydantic v2)
# ------------------------------------------------------------------------------
class Settings(BaseSettings):
    """Strongly typed application configuration settings."""

    llm_provider: Literal["gemini", "openai"] = Field(
        default="gemini",
        validation_alias="LLM_PROVIDER",
        description="The active LLM vendor ('gemini' or 'openai')",
    )

    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
        description="Google AI Studio API Key",
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI Platform API Key",
    )

    llm_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias="LLM_MODEL",
        description="Default target model name",
    )

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        validation_alias="TEMPERATURE",
        description="Sampling temperature for LLM generation",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: Any) -> str:
        """Normalizes and strips provider name to lowercase."""
        if isinstance(value, str):
            return value.strip().lower()
        return "gemini"

    def validate_active_credentials(self) -> None:
        """
        Validates that the active provider has a usable, non-placeholder API key.
        Raises ConfigurationError with actionable remediation steps.
        """
        placeholders = {
            "your_gemini_api_key_here",
            "your_openai_api_key_here",
            "none",
            "",
        }

        if self.llm_provider == "gemini":
            key = (self.gemini_api_key or "").strip()
            if not key or key.lower() in placeholders:
                _report_missing_key_error(
                    provider_label="Google Gemini",
                    provider_code="gemini",
                    env_var="GEMINI_API_KEY",
                    signup_url="https://aistudio.google.com/",
                )
                raise ConfigurationError(
                    "Missing or invalid GEMINI_API_KEY. Please update your .env file."
                )

        elif self.llm_provider == "openai":
            key = (self.openai_api_key or "").strip()
            if not key or key.lower() in placeholders:
                _report_missing_key_error(
                    provider_label="OpenAI",
                    provider_code="openai",
                    env_var="OPENAI_API_KEY",
                    signup_url="https://platform.openai.com/api-keys",
                )
                raise ConfigurationError(
                    "Missing or invalid OPENAI_API_KEY. Please update your .env file."
                )


# ------------------------------------------------------------------------------
# 4. Diagnostic & Error Reporting Utilities
# ------------------------------------------------------------------------------
def _report_missing_key_error(
    provider_label: str,
    provider_code: str,
    env_var: str,
    signup_url: str,
) -> None:
    """Renders a friendly diagnostic panel explaining missing credential remediation."""
    panel_content = (
        f"[bold red]Authentication Error: Missing API Key for {provider_label}[/bold red]\n\n"
        f"The environment variable [bold yellow]{env_var}[/bold yellow] is required "
        f"when [bold cyan]LLM_PROVIDER={provider_code}[/bold cyan].\n\n"
        f"[bold green]Quick Fix Steps:[/bold green]\n"
        f"  1. Ensure you have copied [dim].env.example[/dim] to [dim].env[/dim]:\n"
        f"     [dim cyan]copy summarizer\\.env.example summarizer\\.env[/dim cyan] (Windows)\n"
        f"     [dim cyan]cp summarizer/.env.example summarizer/.env[/dim cyan] (Unix)\n"
        f"  2. Obtain an API key from: [link={signup_url}]{signup_url}[/link]\n"
        f"  3. Open [bold].env[/bold] and set:\n"
        f"     [bold]{env_var}=AIzaSy...[/bold]\n"
    )
    console.print(Panel(panel_content, title="Configuration Missing", border_style="red"))


# ------------------------------------------------------------------------------
# 5. Client Wrapper & Factory
# ------------------------------------------------------------------------------
class LLMClientWrapper:
    """
    Modular wrapper exposing an authenticated client instance with unified metadata
    and streamlined text inference across Gemini and OpenAI SDKs.
    """

    def __init__(
        self,
        client: Any,
        provider: Literal["gemini", "openai"],
        model: str,
        temperature: float,
    ):
        self.raw_client = client
        self.provider = provider
        self.model = model
        self.temperature = temperature

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Executes a synchronous completion call using the native SDK."""
        if self.provider == "gemini":
            config = genai_types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=system_instruction if system_instruction else None,
            )
            response = self.raw_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        if self.provider == "openai":
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = self.raw_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""

        raise ValueError(f"Unsupported provider: {self.provider}")


def get_settings() -> Settings:
    """Instantiates and returns application settings."""
    return Settings()


def get_llm_client(settings: Optional[Settings] = None) -> LLMClientWrapper:
    """
    Factory function:
      1. Validates configuration and credentials.
      2. Initializes the vendor-specific SDK client.
      3. Returns an initialized LLMClientWrapper ready for inference calls.
    """
    if settings is None:
        settings = get_settings()

    settings.validate_active_credentials()

    if settings.llm_provider == "gemini":
        client = genai.Client(api_key=settings.gemini_api_key)
        return LLMClientWrapper(
            client=client,
            provider="gemini",
            model=settings.llm_model,
            temperature=settings.temperature,
        )

    if settings.llm_provider == "openai":
        client = OpenAI(api_key=settings.openai_api_key)
        return LLMClientWrapper(
            client=client,
            provider="openai",
            model=settings.llm_model,
            temperature=settings.temperature,
        )

    raise ConfigurationError(f"Unsupported provider: {settings.llm_provider}")


# ------------------------------------------------------------------------------
# 6. Self-Diagnostic Runner
# ------------------------------------------------------------------------------
def run_diagnostics() -> None:
    """Executes a self-diagnostic check on loaded settings and client initialization."""
    console.print(
        Panel.fit(
            "[bold cyan]AI Document & Text Summarizer[/bold cyan]\n"
            "[dim]Task 1: Environment & Client Loader Diagnostic[/dim]",
            border_style="cyan",
        )
    )

    try:
        app_settings = get_settings()
        console.print(
            f"[bold green][OK][/bold green] Loaded configuration from: "
            f"[dim]{_ENV_PATH}[/dim]"
        )
        console.print(
            f"[bold green][OK][/bold green] Active Provider: "
            f"[bold]{app_settings.llm_provider.upper()}[/bold]"
        )
        console.print(
            f"[bold green][OK][/bold green] Target Model: "
            f"[bold]{app_settings.llm_model}[/bold]"
        )
        console.print(
            f"[bold green][OK][/bold green] Temperature: "
            f"[bold]{app_settings.temperature}[/bold]"
        )

        wrapper = get_llm_client(app_settings)
        console.print(
            Panel(
                f"[bold green]LLM Client successfully initialized![/bold green]\n"
                f"Provider: {wrapper.provider}\n"
                f"Model: {wrapper.model}\n"
                f"Client Instance: {wrapper.raw_client.__class__.__name__}",
                title="Initialization Success",
                border_style="green",
            )
        )
    except ConfigurationError as exc:
        console.print(f"[bold red]Setup Required:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    run_diagnostics()
