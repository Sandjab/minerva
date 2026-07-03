"""Backend d'extraction via une API compatible OpenAI (OpenAI, Ollama, etc.)."""

from __future__ import annotations

import os

import openai
from pydantic import ValidationError

from . import ExtractionResult

DEFAULT_MODEL = "gpt-4o"


class OpenAIBackend:
    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        # Pour Ollama : base_url="http://localhost:11434/v1" ; la clé est
        # ignorée par le serveur mais exigée par le SDK.
        api_key = os.environ.get("OPENAI_API_KEY") or ("ollama" if base_url else None)
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._model = model or DEFAULT_MODEL

    def extract(self, system: str, user: str) -> ExtractionResult:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": ExtractionResult.model_json_schema(),
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Réponse vide du modèle")
        try:
            return ExtractionResult.model_validate_json(content)
        except ValidationError as exc:
            raise RuntimeError(f"Réponse non conforme au schéma : {exc}") from exc
