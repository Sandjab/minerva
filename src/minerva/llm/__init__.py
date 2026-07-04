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


# --- Schéma de la passe unique enrichie (extraction + timeline) ---------------
# Mêmes contraintes structured outputs : pas de dict libre, pas d'Optional —
# les champs absents sont des chaînes vides / 0.


class ExtractedFact(BaseModel):
    """Fait rattaché à un moment : attribut d'entité (entity+attribute+value)
    OU relation (relation+source+target, attribute/value optionnels)."""

    entity: str = ""
    relation: str = ""
    source: str = ""
    target: str = ""
    attribute: str = ""
    value: str = ""


class ExtractedTransition(BaseModel):
    type: str = "suite"     # suite | ellipse | flashback | retour | parallèle
    target: str = ""        # identifiant d'un moment connu (retour/parallèle)
    gap_text: str = ""      # expression du texte (« vingt ans après »)
    gap_value: float = 0    # écart quantifié, 0 = non quantifié
    gap_unit: str = ""      # heures | jours | semaines | mois | années


class ExtractedMoment(BaseModel):
    ref: str = ""           # identifiant local au chunk (« m1 », « m2 »)
    summary: str = ""
    transition: ExtractedTransition = Field(default_factory=ExtractedTransition)
    entities_present: list[str] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


class ExtractedIdentity(BaseModel):
    """Entité côté identité seule : les attributs passent par les facts."""

    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)


class TimelineExtractionResult(BaseModel):
    entities: list[ExtractedIdentity] = Field(default_factory=list)
    moments: list[ExtractedMoment] = Field(default_factory=list)


class LLMBackend(Protocol):
    """Backend LLM à sortie structurée générique.

    `parse` renvoie une instance validée de `output_model` (n'importe quel
    modèle Pydantic : extraction, arbitrage, etc.).
    """

    def parse(self, system: str, user: str, output_model: type[T]) -> T: ...


def make_backend(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
) -> LLMBackend:
    """Fabrique un backend d'extraction.

    provider : "anthropic" ou "openai" (ce dernier couvre Ollama et tout
    serveur compatible OpenAI via base_url).
    """
    if provider == "anthropic":
        if temperature is not None:
            raise ValueError(
                "temperature n'est pas configurable sur le backend anthropic "
                "(les modèles Claude récents rejettent ce paramètre)"
            )
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model)
    if provider == "openai":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(model=model, base_url=base_url, temperature=temperature)
    raise ValueError(f"Provider inconnu : {provider!r} (attendu : anthropic ou openai)")
