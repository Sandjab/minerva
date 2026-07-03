"""Backend d'extraction via l'API Claude (SDK anthropic officiel)."""

from __future__ import annotations

import anthropic

from . import ExtractionResult

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicBackend:
    def __init__(self, model: str | None = None) -> None:
        # Le client résout les identifiants depuis l'environnement
        # (ANTHROPIC_API_KEY, ou un profil `ant auth login`).
        self._client = anthropic.Anthropic()
        self._model = model or DEFAULT_MODEL

    def extract(self, system: str, user: str) -> ExtractionResult:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=ExtractionResult,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(
                f"Réponse non conforme au schéma (stop_reason={response.stop_reason})"
            )
        return parsed
