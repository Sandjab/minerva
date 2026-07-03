"""Backend à sortie structurée via une API compatible OpenAI (OpenAI, Ollama, etc.)."""

from __future__ import annotations

import os
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "gpt-4o"

T = TypeVar("T", bound=BaseModel)


class OpenAIBackend:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        # Pour Ollama : base_url="http://localhost:11434/v1" ; la clé est
        # ignorée par le serveur mais exigée par le SDK.
        api_key = os.environ.get("OPENAI_API_KEY") or ("ollama" if base_url else None)
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._model = model or DEFAULT_MODEL
        self._temperature = temperature

    def parse(self, system: str, user: str, output_model: type[T]) -> T:
        kwargs = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__.lower(),
                    "schema": output_model.model_json_schema(),
                },
            },
            **kwargs,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Réponse vide du modèle")
        try:
            return output_model.model_validate_json(content)
        except ValidationError as exc:
            raise RuntimeError(f"Réponse non conforme au schéma : {exc}") from exc
