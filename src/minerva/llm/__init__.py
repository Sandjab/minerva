"""Backends LLM et schéma de sortie structurée de l'extraction.

Contrainte des structured outputs : tout objet du JSON Schema doit porter
``additionalProperties: false`` — un dict libre n'est pas exprimable. Les
attributs dynamiques sont donc transportés comme listes de paires
``{name, value}`` puis convertis en dict côté code.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, Field

from ..model import Entity, Relation

T = TypeVar("T", bound=BaseModel)


class ExtractedAttribute(BaseModel):
    name: str
    value: str


class ExtractedEntity(BaseModel):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    attributes: list[ExtractedAttribute] = Field(default_factory=list)

    def to_entity(self) -> Entity:
        return Entity(
            name=self.name,
            type=self.type,
            aliases=self.aliases,
            attributes={a.name: a.value for a in self.attributes},
        )


class ExtractedRelation(BaseModel):
    name: str
    source: str
    target: str
    attributes: list[ExtractedAttribute] = Field(default_factory=list)

    def to_relation(self) -> Relation:
        return Relation(
            name=self.name,
            source=self.source,
            target=self.target,
            attributes={a.name: a.value for a in self.attributes},
        )


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class LLMBackend(Protocol):
    """Backend LLM à sortie structurée générique.

    `parse` renvoie une instance validée de `output_model` (n'importe quel
    modèle Pydantic : extraction, arbitrage, etc.).
    """

    def parse(self, system: str, user: str, output_model: type[T]) -> T: ...


def make_backend(
    provider: str, model: str | None = None, base_url: str | None = None
) -> LLMBackend:
    """Fabrique un backend d'extraction.

    provider : "anthropic" ou "openai" (ce dernier couvre Ollama et tout
    serveur compatible OpenAI via base_url).
    """
    if provider == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model)
    if provider == "openai":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(model=model, base_url=base_url)
    raise ValueError(f"Provider inconnu : {provider!r} (attendu : anthropic ou openai)")
