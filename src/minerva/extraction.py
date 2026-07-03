"""Orchestration de l'extraction : chunking, appels LLM, fusion incrémentale."""

from __future__ import annotations

from collections.abc import Callable

from .chunking import DEFAULT_CHUNK_SIZE, split_text
from .llm import ExtractionResult, LLMBackend
from .model import Entity, KnowledgeGraph, Relation

MAX_KNOWN_ENTITIES_IN_PROMPT = 200

SYSTEM_PROMPT = """\
Tu es un extracteur d'informations. On te donne un extrait de texte (par exemple \
d'un roman). Extrais-en toutes les entités nommées et toutes les relations entre \
elles, avec leurs attributs.

Règles :
- `name` d'une entité : son nom canonique le plus complet dans le texte \
(ex. « Jean Valjean », pas « il »). Mets les autres désignations dans `aliases`.
- `type` : catégorie libre en français, en minuscules (ex. « personnage », \
« lieu », « objet », « organisation »).
- `attributes` : toute propriété affirmée par le texte (ex. « âge » : « 25 ans », \
« profession » : « forçat »). Noms d'attributs en français, en minuscules. \
N'invente rien : uniquement ce que le texte dit.
- Une relation relie deux entités par un nom libre en français (ex. « aime », \
« père de », « habite »). `source` et `target` sont des noms canoniques \
d'entités. Les attributs d'une relation qualifient la relation elle-même \
(ex. « depuis » : « l'enfance »).
- Si une liste d'entités déjà connues est fournie, réutilise exactement ces \
noms canoniques quand l'extrait parle des mêmes entités.\
"""


def build_user_prompt(chunk: str, known_entities: list[str]) -> str:
    parts = []
    if known_entities:
        names = "\n".join(f"- {n}" for n in known_entities[:MAX_KNOWN_ENTITIES_IN_PROMPT])
        parts.append(f"Entités déjà connues :\n{names}")
    parts.append(f"Extrait à analyser :\n\n{chunk}")
    return "\n\n".join(parts)


def sanitize(result: ExtractionResult) -> tuple[list[Entity], list[Relation]]:
    """Écarte le bruit LLM : entités sans nom, relations incomplètes."""
    entities = [e.to_entity() for e in result.entities if e.name.strip()]
    relations = [
        r.to_relation()
        for r in result.relations
        if r.name.strip() and r.source.strip() and r.target.strip()
    ]
    return entities, relations


def extract_graph(
    text: str,
    backend: LLMBackend,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
) -> KnowledgeGraph:
    """Extrait le graphe complet d'un texte, chunk par chunk."""
    graph = KnowledgeGraph()
    chunks = split_text(text, chunk_size)
    for i, chunk in enumerate(chunks):
        known = [e.name for e in graph.entities]
        result = backend.extract(SYSTEM_PROMPT, build_user_prompt(chunk, known))
        graph.merge(*sanitize(result))
        if on_progress:
            on_progress(i + 1, len(chunks))
    return graph
