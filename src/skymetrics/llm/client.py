"""Minimal OpenRouter chat client using free models.

Reads ``OPENROUTER_API_KEY`` from the environment. The HTTP POST is injectable
so the client is unit-tested without any network call or API cost.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free-tier model on OpenRouter (no credit cost).
DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

Poster = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


class LLMConfigError(RuntimeError):
    """Raised when the client is used without an API key."""


def _default_poster(url: str, json_body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    import requests  # noqa: PLC0415

    resp = requests.post(url, json=json_body, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


class OpenRouterClient:
    """Thin wrapper over the OpenRouter chat-completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        poster: Poster | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self._poster = poster or _default_poster

    def complete(self, prompt: str, system: str = "You are a helpful analyst.") -> str:
        """Return the model's reply for a single user prompt.

        Raises:
            LLMConfigError: If no API key is configured.
        """
        if not self.api_key:
            raise LLMConfigError("OPENROUTER_API_KEY is not set")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = self._poster(OPENROUTER_URL, body, headers)
        return data["choices"][0]["message"]["content"]
