"""Fusion de graphes extraits par plusieurs modèles (ensemblisme).

Stratégies :
- union : tout fusionner (rappel maximal, bruit maximal) ;
- vote : garder les items présents dans >= min_votes graphes ;
- arbitrated : consensus (vote) gardé d'office, items minoritaires tranchés
  par un LLM arbitre qui relit le texte source.

Le vote des entités passe par la résolution du graphe (alias, titres,
Unicode). Le vote des relations porte sur la paire orientée (source, cible) :
les modèles paraphrasent les prédicats (« protège » vs « veille sur »), donc
exiger le même nom de relation sous-compterait l'accord ; tous les prédicats
distincts d'une paire retenue sont conservés.

L'ordre des graphes fait priorité : en cas de conflit d'attribut ou de nom
canonique, le graphe le plus tôt dans la liste gagne (mettre le meilleur
modèle en premier).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import LLMBackend
from .model import KnowledgeGraph, Relation, normalize


def merge_union(graphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    merged = KnowledgeGraph()
    for graph in graphs:
        merged.merge(graph.entities, graph.relations)
    return merged


def _entity_votes(union: KnowledgeGraph, graphs: list[KnowledgeGraph]) -> dict[str, int]:
    """Nombre de graphes contenant chaque entité canonique de l'union."""
    votes: dict[str, int] = {normalize(e.name): 0 for e in union.entities}
    for graph in graphs:
        seen: set[str] = set()
        for entity in graph.entities:
            canonical = union.resolve(entity.name)
            if canonical is not None:
                seen.add(normalize(canonical.name))
        for key in seen:
            votes[key] = votes.get(key, 0) + 1
    return votes


def _relation_votes(
    union: KnowledgeGraph, graphs: list[KnowledgeGraph]
) -> dict[tuple[str, str], int]:
    """Nombre de graphes reliant chaque paire orientée (source, cible)."""
    votes: dict[tuple[str, str], int] = {}
    for graph in graphs:
        seen: set[tuple[str, str]] = set()
        for rel in graph.relations:
            src = union.resolve(rel.source)
            tgt = union.resolve(rel.target)
            if src is None or tgt is None:
                continue
            seen.add((normalize(src.name), normalize(tgt.name)))
        for pair in seen:
            votes[pair] = votes.get(pair, 0) + 1
    return votes


def _pair_of(union: KnowledgeGraph, rel: Relation) -> tuple[str, str] | None:
    src = union.resolve(rel.source)
    tgt = union.resolve(rel.target)
    if src is None or tgt is None:
        return None
    return (normalize(src.name), normalize(tgt.name))


def _build(
    union: KnowledgeGraph,
    kept_entities: set[str],
    kept_pairs: set[tuple[str, str]],
) -> KnowledgeGraph:
    """Reconstruit un graphe restreint aux entités et paires retenues."""
    result = KnowledgeGraph()
    for entity in union.entities:
        if normalize(entity.name) in kept_entities:
            result.add_entity(entity.model_copy(deep=True))
    for rel in union.relations:
        pair = _pair_of(union, rel)
        if pair is None or pair not in kept_pairs:
            continue
        if pair[0] in kept_entities and pair[1] in kept_entities:
            result.add_relation(rel.model_copy(deep=True))
    return result


def merge_vote(graphs: list[KnowledgeGraph], min_votes: int = 2) -> KnowledgeGraph:
    union = merge_union(graphs)
    entity_votes = _entity_votes(union, graphs)
    relation_votes = _relation_votes(union, graphs)
    kept_entities = {k for k, v in entity_votes.items() if v >= min_votes}
    kept_pairs = {p for p, v in relation_votes.items() if v >= min_votes}
    return _build(union, kept_entities, kept_pairs)


# --- Arbitrage LLM des items minoritaires -------------------------------------

ARBITER_SYSTEM = """\
Tu es un vérificateur d'extraction d'information. Plusieurs modèles ont extrait \
des entités et relations d'un texte ; les items ci-dessous n'ont PAS fait \
consensus (présents chez une minorité de modèles). Pour chaque item, relis le \
texte source et décide s'il faut le garder (`keep: true` s'il est réellement \
soutenu par le texte : entité nommée légitime, relation affirmée) ou l'écarter \
(`keep: false` : invention, redite d'une autre entité, fragment, périphrase qui \
n'est pas une entité). Rends une décision pour CHAQUE id fourni.\
"""


class ArbiterDecision(BaseModel):
    id: str
    keep: bool


class ArbiterResult(BaseModel):
    decisions: list[ArbiterDecision] = Field(default_factory=list)


def merge_arbitrated(
    graphs: list[KnowledgeGraph],
    arbiter: LLMBackend,
    text: str,
    min_votes: int = 2,
) -> KnowledgeGraph:
    """Consensus gardé d'office ; items minoritaires tranchés par l'arbitre.

    L'arbitre ne voit que les items en litige et le texte source — les items
    consensuels ne consomment pas de tokens d'arbitrage.
    """
    union = merge_union(graphs)
    entity_votes = _entity_votes(union, graphs)
    relation_votes = _relation_votes(union, graphs)

    kept_entities = {k for k, v in entity_votes.items() if v >= min_votes}
    kept_pairs = {p for p, v in relation_votes.items() if v >= min_votes}

    disputed_entities = [
        e for e in union.entities if normalize(e.name) not in kept_entities
    ]
    disputed_pairs = sorted(
        {p for p, v in relation_votes.items() if v < min_votes}
    )

    lines: list[str] = []
    ids: dict[str, tuple[str, object]] = {}
    for i, entity in enumerate(disputed_entities):
        item_id = f"E{i}"
        ids[item_id] = ("entity", normalize(entity.name))
        attrs = ", ".join(f"{k}={v}" for k, v in entity.attributes.items())
        lines.append(f"- {item_id} [entité] {entity.name} ({entity.type}){' — ' + attrs if attrs else ''}")
    for i, pair in enumerate(disputed_pairs):
        item_id = f"R{i}"
        ids[item_id] = ("pair", pair)
        names = [r.name for r in union.relations if _pair_of(union, r) == pair]
        lines.append(f"- {item_id} [relation] {pair[0]} -> {pair[1]} ({' / '.join(names)})")

    if not lines:
        return _build(union, kept_entities, kept_pairs)

    user = (
        "Texte source :\n\n" + text + "\n\nItems en litige :\n" + "\n".join(lines)
    )
    result = arbiter.parse(ARBITER_SYSTEM, user, ArbiterResult)
    decided = {d.id: d.keep for d in result.decisions}

    for item_id, (kind, key) in ids.items():
        if not decided.get(item_id, False):  # non tranché = écarté (prudence)
            continue
        if kind == "entity":
            kept_entities.add(key)  # type: ignore[arg-type]
        else:
            kept_pairs.add(key)  # type: ignore[arg-type]

    # Une relation gardée exige ses deux extrémités : on repêche les entités
    # d'une paire gardée même si l'entité seule avait été écartée.
    for pair in kept_pairs:
        kept_entities.update(pair)

    return _build(union, kept_entities, kept_pairs)
