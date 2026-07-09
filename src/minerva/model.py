"""Modèle de données : entités nommées, relations nommées, attributs dynamiques."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from .timeline import Moment, Timeline

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


class Assertion(BaseModel):
    """Constat bitemporel : un fait affirmé par le texte, ancré à un moment
    diégétique (moment_id, None = non daté) et à sa position de lecture
    (chunk_index). Sujet : une entité (entity) OU une relation (triplet)."""

    entity: str = ""
    relation_name: str = ""
    relation_source: str = ""
    relation_target: str = ""
    attribute: str = ""
    value: str = ""
    moment_id: int | None = None
    chunk_index: int | None = None

    def key(self) -> tuple:
        return (
            normalize(self.entity), normalize(self.relation_name),
            normalize(self.relation_source), normalize(self.relation_target),
            self.attribute, self.value, self.moment_id,
        )


class KnowledgeGraph:
    """Graphe d'entités et de relations, avec fusion déterministe.

    Règle de fusion : les valeurs déjà présentes sont préservées (première
    extraction gagne) ; les attributs et alias nouveaux sont ajoutés.

    Cette règle décrit les dicts `Entity.attributes` / `Relation.attributes`,
    mais ce ne sont qu'une VUE « première extraction gagne » maintenue en
    écriture par le chemin unique `add_assertion`. La source de vérité est le
    journal de constats (`assertions`), bitemporel : voir `entity_state` /
    `relation_state` pour dériver d'autres vues (dernière valeur en ordre
    diégétique résolu, snapshot à un moment donné via `at=`).
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}  # clé = nom canonique normalisé
        self._alias_index: dict[str, str] = {}  # alias normalisé -> clé canonique
        # forme sans titre/article -> clé canonique, ou None si ambiguë
        # (ex. « M. Thénardier » et « Mme Thénardier » donnent la même forme)
        self._stripped_index: dict[str, str | None] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}
        self.timeline = Timeline()
        self._assertions: list[Assertion] = []
        self._assertion_keys: set[tuple] = set()

    @property
    def entities(self) -> list[Entity]:
        return list(self._entities.values())

    @property
    def relations(self) -> list[Relation]:
        return list(self._relations.values())

    @property
    def assertions(self) -> list[Assertion]:
        return list(self._assertions)

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
            # Type normalisé en minuscules à l'écriture (convention « types en
            # minuscules ») : le sentinel « inconnu » devient insensible à la
            # casse et les variantes (« Personnage ») sont repliées.
            stored = Entity(
                name=entity.name.strip(),
                type=entity.type.strip().lower() or "inconnu",
            )
            self._entities[key] = stored
            self._index_stripped(key, key)
            existing = stored
        elif existing.type == "inconnu":
            # « inconnu » est le défaut des chemins d'auto-création (extrémité
            # de relation, sujet d'assertion), pas une valeur extraite : un vrai
            # type le comble. Deux vrais types en conflit -> première extraction
            # typée gagne (on n'écrase jamais un type déjà réel).
            incoming = entity.type.strip().lower()
            if incoming and incoming != "inconnu":
                existing.type = incoming
        self._register_aliases(existing, entity.aliases)
        for attr, value in entity.attributes.items():
            self.add_assertion(Assertion(entity=existing.name, attribute=attr, value=value))
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
        )
        existing = self._relations.get(resolved.key())
        if existing is None:
            self._relations[resolved.key()] = resolved
            existing = resolved
        for attr, value in relation.attributes.items():
            self.add_assertion(
                Assertion(relation_name=existing.name, relation_source=existing.source,
                          relation_target=existing.target, attribute=attr, value=value)
            )
        return existing

    def merge(self, entities: list[Entity], relations: list[Relation]) -> None:
        for entity in entities:
            self.add_entity(entity)
        for relation in relations:
            self.add_relation(relation)

    def add_assertion(self, assertion: Assertion) -> Assertion | None:
        """Chemin d'écriture unique du journal. Résout les sujets vers leurs
        noms canoniques (création au besoin), déduplique sur
        (sujet, attribut, valeur, moment), et maintient la vue
        « première extraction gagne » des dicts d'attributs."""
        a = assertion.model_copy()
        a.attribute = a.attribute.strip()
        a.value = a.value.strip()
        if a.entity.strip():
            if not a.attribute:
                return None  # la présence seule passe par timeline.add_appearance
            entity = self.resolve(a.entity) or self.add_entity(
                Entity(name=a.entity, type="inconnu")
            )
            a.entity = entity.name
            view = entity.attributes
        elif a.relation_name.strip() and a.relation_source.strip() and a.relation_target.strip():
            relation = self.add_relation(
                Relation(name=a.relation_name, source=a.relation_source,
                         target=a.relation_target)
            )
            a.relation_name, a.relation_source, a.relation_target = (
                relation.name, relation.source, relation.target
            )
            view = relation.attributes if a.attribute else None
        else:
            return None
        if a.moment_id is not None and self.timeline.moment(a.moment_id) is None:
            a.moment_id = None  # référence de moment invalide : constat non daté
        if a.key() in self._assertion_keys:
            return None
        self._assertion_keys.add(a.key())
        self._assertions.append(a)
        if view is not None and a.attribute:
            view.setdefault(a.attribute, a.value)
        return a

    def entity_state(
        self, policy: str = "first", at: int | None = None
    ) -> dict[str, dict[str, str]]:
        """Snapshot {entité: {attribut: valeur}}. « first » = première
        extraction gagne (comportement historique) ; « final » = dernière
        valeur en ordre diégétique résolu (non daté = avant tout).

        `at` (id de moment, uniquement avec policy="final") restreint le
        snapshot à ce qu'on sait au moment `at` inclus : seules les
        assertions dont le rang est <= celui de `at` sont considérées (les
        assertions non datées, toujours de rang -1, restent incluses)."""
        if policy == "first":
            if at is not None:
                raise ValueError("at= exige policy='final'")
            return {e.name: dict(e.attributes) for e in self.entities}
        if policy != "final":
            raise ValueError(f"policy inconnue : {policy!r}")
        rank = self._moment_ranks()
        at_rank = self._at_rank(at, rank)
        best: dict[tuple[str, str], tuple[int, str]] = {}
        for a in self._assertions:
            if not (a.entity and a.attribute):
                continue
            r = rank.get(a.moment_id, -1)
            if at_rank is not None and r > at_rank:
                continue
            k = (a.entity, a.attribute)
            if k not in best or r > best[k][0]:
                best[k] = (r, a.value)
        state: dict[str, dict[str, str]] = {e.name: {} for e in self.entities}
        for (name, attr), (_, value) in best.items():
            state.setdefault(name, {})[attr] = value
        return state

    def relation_state(
        self, policy: str = "first", at: int | None = None
    ) -> dict[tuple[str, str, str], dict[str, str]]:
        """Snapshot {(nom, source, cible): {attribut: valeur}} des relations.
        `at` : voir entity_state."""
        if policy == "first":
            if at is not None:
                raise ValueError("at= exige policy='final'")
            return {(r.name, r.source, r.target): dict(r.attributes) for r in self.relations}
        if policy != "final":
            raise ValueError(f"policy inconnue : {policy!r}")
        rank = self._moment_ranks()
        at_rank = self._at_rank(at, rank)
        best: dict[tuple, tuple[int, str]] = {}
        for a in self._assertions:
            if a.entity or not a.attribute or not a.relation_name:
                continue
            r = rank.get(a.moment_id, -1)
            if at_rank is not None and r > at_rank:
                continue
            k = ((a.relation_name, a.relation_source, a.relation_target), a.attribute)
            if k not in best or r > best[k][0]:
                best[k] = (r, a.value)
        state = {(r.name, r.source, r.target): {} for r in self.relations}
        for (triple, attr), (_, value) in best.items():
            state.setdefault(triple, {})[attr] = value
        return state

    def _moment_ranks(self) -> dict[int | None, int]:
        """resolved_order par moment ; None (non daté) -> -1 (avant tout).
        À égalité de rang, la première assertion gagne (déterministe)."""
        self.timeline.resolve()
        ranks: dict[int | None, int] = {None: -1}
        for m in self.timeline.moments:
            ranks[m.id] = m.resolved_order if m.resolved_order is not None else -1
        return ranks

    def _at_rank(self, at: int | None, rank: dict[int | None, int]) -> int | None:
        """Rang du moment `at`, ou None si `at` n'est pas fourni. ValueError
        si `at` ne désigne aucun moment existant."""
        if at is None:
            return None
        if at not in rank:
            raise ValueError(f"moment inexistant : {at!r}")
        return rank[at]

    def track(self, name: str) -> list[tuple[Moment, list[Assertion]]]:
        """Piste d'une entité : ses moments (présence ou constat), en ordre
        diégétique résolu, avec les constats qui la concernent."""
        entity = self.resolve(name)
        if entity is None:
            return []
        self.timeline.resolve()
        by_moment: dict[int, list[Assertion]] = {}
        involved_ids: set[int] = set()
        for a in self._assertions:
            if a.moment_id is None:
                continue
            if a.entity == entity.name or entity.name in (a.relation_source, a.relation_target):
                by_moment.setdefault(a.moment_id, []).append(a)
                involved_ids.add(a.moment_id)
        for mid, names in self.timeline.appearances.items():
            if entity.name in names:
                involved_ids.add(mid)
        moments = sorted(
            (self.timeline.moment(mid) for mid in involved_ids),
            key=lambda m: (m.resolved_order or 0, m.reading_key()),
        )
        return [(m, by_moment.get(m.id, [])) for m in moments]

    def adopt_journal(self, other: "KnowledgeGraph") -> None:
        """Reprend le journal d'un autre graphe en résolvant les sujets dans
        CE graphe (usage : canonicalisation, qui reconstruit les identités).
        Le graphe destination ne doit pas déjà porter de journal."""
        if self.timeline.moments or self._assertions:
            raise ValueError("adopt_journal exige un graphe destination sans journal")
        self.timeline = other.timeline.clone()
        all_names = {n for names in other.timeline.appearances.values() for n in names}
        for n in all_names:
            resolved = self.resolve(n)
            if resolved is not None and resolved.name != n:
                self.timeline.rename_entity(n, resolved.name)
        for a in other.assertions:
            self.add_assertion(a)

    def to_dict(self) -> dict:
        return {
            "entities": [e.model_dump() for e in self.entities],
            "relations": [r.model_dump() for r in self.relations],
            "moments": [m.model_dump() for m in self.timeline.moments],
            "constraints": [c.model_dump() for c in self.timeline.constraints],
            "appearances": {str(k): sorted(v) for k, v in self.timeline.appearances.items()},
            "assertions": [a.model_dump() for a in self._assertions],
        }
