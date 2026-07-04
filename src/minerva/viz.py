"""Restitution graphique : payload prêt à rendre + page HTML autonome.

Toutes les transformations (pistes, écarts, visibilité par moment) se font
ici, en Python — le modèle fait foi ; le JavaScript de la page ne fait que
du rendu. Visibilité par premier rang : rang d'un moment = resolved_order,
trace non datée = -1 (« avant tout », même convention que _moment_ranks)."""

from __future__ import annotations

import json
from importlib.resources import files

from .model import KnowledgeGraph
from .timeline import AVANT


def build_payload(graph: KnowledgeGraph) -> dict:
    graph.timeline.resolve()
    moments = sorted(graph.timeline.moments, key=lambda m: m.resolved_order or 0)
    rank = {m.id: m.resolved_order or 0 for m in moments}

    rel_asserts: dict[tuple[str, str, str], list[dict]] = {}
    for a in graph.assertions:
        if not a.entity and a.relation_name and a.attribute:
            key = (a.relation_name, a.relation_source, a.relation_target)
            rel_asserts.setdefault(key, []).append(
                {"attribute": a.attribute, "value": a.value, "moment_id": a.moment_id}
            )

    gaps: dict[str, float] = {}
    for c in graph.timeline.constraints:
        if c.relation != AVANT or c.gap.days is None:
            continue
        if c.source_id not in rank or c.target_id not in rank:
            continue
        if rank[c.target_id] == rank[c.source_id] + 1:
            gaps[str(rank[c.source_id])] = c.gap.days

    return {
        "entities": [
            {"name": e.name, "type": e.type, "aliases": e.aliases,
             "attributes": e.attributes}
            for e in graph.entities
        ],
        "relations": [
            {"name": r.name, "source": r.source, "target": r.target,
             "attributes": r.attributes,
             "assertions": rel_asserts.get((r.name, r.source, r.target), [])}
            for r in graph.relations
        ],
        "moments": [
            {"id": m.id, "order": rank[m.id], "days": m.resolved_days,
             "summary": m.summary}
            for m in moments
        ],
        "gaps": gaps,
        "tracks": [],   # Task 3
        "states": [],   # Task 4
    }
