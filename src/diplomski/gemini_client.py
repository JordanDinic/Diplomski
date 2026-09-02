from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import errors

from diplomski.settings import (
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    DEFAULT_GEMINI_MODEL,
)


class GeminiFlashClient:
    """Small wrapper around the Google Gen AI SDK."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        max_output_tokens: int = DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    ) -> None:
        _load_environment()

        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.max_output_tokens = max_output_tokens

        if not self.api_key or self.api_key.startswith("your_gemini_api_key_here"):
            raise ValueError(
                "Gemini API key is missing or still contains the placeholder. "
                "Set GEMINI_API_KEY in .env to a valid Google AI Studio API key."
            )

        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate an answer from Gemini Flash."""

        try:
            interaction = self.client.interactions.create(
                model=self.model_name,
                input=prompt,
                system_instruction=system_instruction,
                generation_config={
                    "max_output_tokens": self.max_output_tokens,
                },
                store=False,
            )
        except errors.ClientError as exc:
            raise RuntimeError(_format_gemini_client_error(exc)) from exc

        text = _interaction_text(interaction)
        if text:
            return text.strip()

        return str(interaction).strip()


def _load_environment() -> None:
    """Load .env from the current project directory when it exists."""

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)


def _format_gemini_client_error(exc: errors.ClientError) -> str:
    message = str(exc)

    if "API_KEY_INVALID" in message or "API key not valid" in message:
        return (
            "Gemini API key is invalid. Replace GEMINI_API_KEY in .env with "
            "a valid key from Google AI Studio."
        )

    if "PERMISSION_DENIED" in message:
        return (
            "Gemini request was denied. Check that the API key has access to "
            "the Gemini API and that billing/project permissions are configured."
        )

    if "no longer available" in message and "gemini-2.5-flash" in message:
        return (
            "The configured Gemini model is no longer available to this API key. "
            "Use gemini-3.6-flash in .env or pass --gemini-model gemini-3.6-flash."
        )

    return f"Gemini API request failed: {message}"


def _interaction_text(interaction: object) -> str | None:
    output_text = getattr(interaction, "output_text", None)
    if output_text:
        return str(output_text)

    outputs = getattr(interaction, "outputs", None)
    if outputs:
        last_output = outputs[-1]
        text = getattr(last_output, "text", None)
        if text:
            return str(text)

    text = getattr(interaction, "text", None)
    if text:
        return str(text)

    return None
