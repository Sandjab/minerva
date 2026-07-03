"""Modèle de données : entités nommées, relations nommées, attributs dynamiques."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

# Tirets typographiques (U+2010..U+2015, U+2212) unifiés vers le tiret ASCII :
# les LLM mélangent parfois les graphies (« Montreuil-sur-Mer » vs U+2011).
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")

_ARTICLES = ("l'", "le ", "la ", "les ")
_TITLES = (
    "m.", "mme", "mlle", "monsieur", "madame", "mademoiselle",
    "monseigneur", "mgr", "inspecteur", "commissaire", "docteur", "dr",
    "professeur", "maître", "évêque", "abbé", "capitaine", "colonel",
    "général", "lieutenant", "sergent", "comte", "comtesse", "baron",
    "baronne", "duc", "duchesse", "prince", "princesse", "sir", "lord", "lady",
)


def normalize(name: str) -> str:
    """Clé de résolution d'une entité : NFKC, tirets/apostrophes unifiés,
    minuscules, espaces réduits."""
    name = unicodedata.normalize("NFKC", name)
    name = name.translate(_DASHES).replace("’", "'")
    return re.sub(r"\s+", " ", name.strip()).casefold()


def strip_title(normalized: str) -> str | None:
    """Forme sans article ni titre de civilité initial, ou None si inchangée.

    Heuristique volontairement limitée à une liste fermée de titres français ;
    utilisée seulement en repli de résolution, jamais pour renommer.
    """
    stripped = normalized
    for article in _ARTICLES:
        if stripped.startswith(article):
            stripped = stripped[len(article):]
            break
    for title in _TITLES:
        if stripped.startswith(title + " "):
            stripped = stripped[len(title) + 1 :]
            break
    stripped = stripped.strip()
    return stripped if stripped and stripped != normalized else None


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
        # forme sans titre/article -> clé canonique, ou None si ambiguë
        # (ex. « M. Thénardier » et « Mme Thénardier » donnent la même forme)
        self._stripped_index: dict[str, str | None] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}

    @property
    def entities(self) -> list[Entity]:
        return list(self._entities.values())

    @property
    def relations(self) -> list[Relation]:
        return list(self._relations.values())

    def resolve(self, name: str) -> Entity | None:
        """Retrouve une entité par nom canonique, alias, ou — en repli —
        par forme débarrassée d'un titre/article (« inspecteur Javert » ↔
        « Javert »), uniquement quand cette forme est non ambiguë."""
        key = normalize(name)
        key = self._alias_index.get(key, key)
        if key in self._entities:
            return self._entities[key]
        # Repli 1 : la requête est la forme nue d'un nom titré stocké
        # (« Javert » -> « inspecteur Javert »).
        target = self._stripped_index.get(key)
        if target is not None:
            return self._entities.get(target)
        # Repli 2 : la requête est titrée et sa forme nue existe telle quelle
        # (« inspecteur Javert » -> « Javert »). Jamais de titré -> titré :
        # « Mme Thénardier » ne doit pas rejoindre « M. Thénardier ».
        stripped = strip_title(key)
        if stripped:
            stripped = self._alias_index.get(stripped, stripped)
            return self._entities.get(stripped)
        return None

    def _index_stripped(self, name_key: str, canonical_key: str) -> None:
        stripped = strip_title(name_key)
        if stripped is None:
            return
        if stripped in self._stripped_index and self._stripped_index[stripped] != canonical_key:
            self._stripped_index[stripped] = None  # ambiguë : repli désactivé
        else:
            self._stripped_index.setdefault(stripped, canonical_key)

    def add_entity(self, entity: Entity) -> Entity:
        if not entity.name.strip():
            raise ValueError("Une entité doit avoir un nom non vide")
        existing = self.resolve(entity.name)
        if existing is None:
            key = normalize(entity.name)
            stored = Entity(
                name=entity.name.strip(),
                type=entity.type.strip() or "inconnu",
                attributes=dict(entity.attributes),
            )
            self._entities[key] = stored
            self._index_stripped(key, key)
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
            self._index_stripped(alias_key, canonical_key)
            entity.aliases.append(alias.strip())

    def add_relation(self, relation: Relation) -> Relation:
        if not (relation.name.strip() and relation.source.strip() and relation.target.strip()):
            raise ValueError("Une relation doit avoir un nom, une source et une cible non vides")
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
