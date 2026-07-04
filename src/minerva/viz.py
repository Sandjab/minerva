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

    tracks = []
    for e in graph.entities:
        orders = sorted({rank[m.id] for m, _ in graph.track(e.name) if m.id in rank})
        if orders:
            tracks.append({"entity": e.name, "count": len(orders), "runs": _runs(orders)})
    tracks.sort(key=lambda t: (-t["count"], t["entity"]))

    ent_rank: dict[str, int | None] = {e.name: None for e in graph.entities}

    def note(name: str, r: int) -> None:
        if name in ent_rank:
            prev = ent_rank[name]
            ent_rank[name] = r if prev is None else min(prev, r)

    for mid, names in graph.timeline.appearances.items():
        for n in names:
            note(n, rank.get(mid, -1))

    rel_keys = [(r.name, r.source, r.target) for r in graph.relations]
    rel_rank: dict[tuple[str, str, str], int | None] = {k: None for k in rel_keys}
    for a in graph.assertions:
        r = rank.get(a.moment_id, -1) if a.moment_id is not None else -1
        if a.entity:
            note(a.entity, r)
        elif a.relation_name:
            k = (a.relation_name, a.relation_source, a.relation_target)
            if k in rel_rank:
                prev = rel_rank[k]
                rel_rank[k] = r if prev is None else min(prev, r)
            note(a.relation_source, r)
            note(a.relation_target, r)

    ent_first = {n: (-1 if r is None else r) for n, r in ent_rank.items()}
    rel_first = {
        k: max(-1 if rr is None else rr,
               ent_first.get(k[1], -1), ent_first.get(k[2], -1))
        for k, rr in rel_rank.items()
    }
    states = [
        {"entities": [n for n, fr in ent_first.items() if fr <= rank[m.id]],
         "relations": [i for i, k in enumerate(rel_keys) if rel_first[k] <= rank[m.id]]}
        for m in moments
    ]

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
        "tracks": tracks,
        "states": states,
    }


def render_html(payload: dict) -> str:
    """Assemble la page autonome : template + lib vendorée + données.

    Le JSON est inséré dans un <script> : les séquences `</` sont échappées
    en `<\\/` pour qu'une valeur contenant `</script>` ne ferme pas le tag."""
    assets = files("minerva").joinpath("viz_assets")
    template = assets.joinpath("template.html").read_text(encoding="utf-8")
    lib = assets.joinpath("force-graph.min.js").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__FORCE_GRAPH_JS__", lib).replace("__MINERVA_DATA__", data)


def _runs(orders: list[int]) -> list[list[int]]:
    """Fusionne une liste triée d'ordres en runs consécutifs [début, fin]."""
    runs: list[list[int]] = []
    for o in orders:
        if runs and o == runs[-1][1] + 1:
            runs[-1][1] = o
        else:
            runs.append([o, o])
    return runs
