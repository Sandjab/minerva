"""Modèle de données : entités nommées, relations nommées, attributs dynamiques."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


def normalize(name: str) -> str:
    """Clé de résolution d'une entité : minuscules, espaces réduits."""
    return re.sub(r"\s+", " ", name.strip()).lower()


class Entity(BaseModel):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)


class Relation(BaseModel):
    name: str
    source: str  # nom canonique de l'entité source
    target: str  # nom canonique de l'entité cible
    attributes: dict[str, str] = Field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        return (normalize(self.name), normalize(self.source), normalize(self.target))


class KnowledgeGraph:
    """Graphe d'entités et de relations, avec fusion déterministe.

    Règle de fusion : les valeurs déjà présentes sont préservées (première
    extraction gagne) ; les attributs et alias nouveaux sont ajoutés.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}  # clé = nom canonique normalisé
        self._alias_index: dict[str, str] = {}  # alias normalisé -> clé canonique
        self._relations: dict[tuple[str, str, str], Relation] = {}

    @property
    def entities(self) -> list[Entity]:
        return list(self._entities.values())

    @property
    def relations(self) -> list[Relation]:
        return list(self._relations.values())

    def resolve(self, name: str) -> Entity | None:
        """Retrouve une entité par nom canonique ou alias."""
        key = normalize(name)
        key = self._alias_index.get(key, key)
        return self._entities.get(key)

    def add_entity(self, entity: Entity) -> Entity:
        existing = self.resolve(entity.name)
        if existing is None:
            key = normalize(entity.name)
            stored = Entity(
                name=entity.name.strip(),
                type=entity.type.strip(),
                attributes=dict(entity.attributes),
            )
            self._entities[key] = stored
            existing = stored
        else:
            for attr, value in entity.attributes.items():
                existing.attributes.setdefault(attr, value)
        self._register_aliases(existing, entity.aliases)
        return existing

    def _register_aliases(self, entity: Entity, aliases: list[str]) -> None:
        canonical_key = normalize(entity.name)
        for alias in aliases:
            alias_key = normalize(alias)
            if not alias_key or alias_key == canonical_key or alias_key in self._alias_index:
                continue
            if alias_key in self._entities:
                continue  # l'alias désigne déjà une autre entité canonique
            self._alias_index[alias_key] = canonical_key
            entity.aliases.append(alias.strip())

    def add_relation(self, relation: Relation) -> Relation:
        # Les extrémités sont résolues (alias -> nom canonique) et créées au besoin.
        source = self.resolve(relation.source) or self.add_entity(
            Entity(name=relation.source, type="inconnu")
        )
        target = self.resolve(relation.target) or self.add_entity(
            Entity(name=relation.target, type="inconnu")
        )
        resolved = Relation(
            name=relation.name.strip(),
            source=source.name,
            target=target.name,
            attributes=dict(relation.attributes),
        )
        existing = self._relations.get(resolved.key())
        if existing is None:
            self._relations[resolved.key()] = resolved
            return resolved
        for attr, value in resolved.attributes.items():
            existing.attributes.setdefault(attr, value)
        return existing

    def merge(self, entities: list[Entity], relations: list[Relation]) -> None:
        for entity in entities:
            self.add_entity(entity)
        for relation in relations:
            self.add_relation(relation)

    def to_dict(self) -> dict:
        return {
            "entities": [e.model_dump() for e in self.entities],
            "relations": [r.model_dump() for r in self.relations],
        }
