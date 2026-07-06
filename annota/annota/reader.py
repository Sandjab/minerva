"""Lecture d'une base minerva → surface forms, partition prédite, contexte.

Dépend du schéma SQL de minerva et de minerva.chunking (même découpage qu'à
l'extraction, pour que assertions.chunk_index pointe sur le bon passage)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from minerva.chunking import DEFAULT_CHUNK_SIZE, split_text


@dataclass(frozen=True)
class SurfaceForm:
    entity_id: int
    kind: str          # 'name' | 'alias'
    surface_form: str
    entity_type: str


def surface_forms(conn: sqlite3.Connection) -> list[SurfaceForm]:
    out: list[SurfaceForm] = []
    for eid, name, etype in conn.execute("SELECT id, name, type FROM entities"):
        out.append(SurfaceForm(eid, "name", name, etype))
    for eid, alias, etype in conn.execute(
        "SELECT a.entity_id, a.alias, e.type FROM entity_aliases a "
        "JOIN entities e ON e.id = a.entity_id"
    ):
        out.append(SurfaceForm(eid, "alias", alias, etype))
    return out


def predicted_partition(conn: sqlite3.Connection) -> dict:
    """{ (entity_id, kind, surface_form) -> str(entity_id) } : toutes les surface
    forms d'une même entité partagent le cluster prédit."""
    return {
        (s.entity_id, s.kind, s.surface_form): str(s.entity_id)
        for s in surface_forms(conn)
    }


def build_chunks(source_text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    return split_text(source_text, chunk_size)


def _entity_forms(conn: sqlite3.Connection, entity_id: int) -> list[str]:
    """Toutes les formes de surface d'une entité (name + aliases), pour repérer
    ses mentions dans le texte."""
    forms = [r[0] for r in conn.execute("SELECT name FROM entities WHERE id = ?", (entity_id,))]
    forms += [r[0] for r in conn.execute(
        "SELECT alias FROM entity_aliases WHERE entity_id = ?", (entity_id,))]
    return forms


def _windows_in_chunk(chunk: str, forms: list[str], window: int) -> list[str]:
    """Extrait des fenêtres de `±window` caractères autour de chaque occurrence
    d'une forme dans `chunk`. Les fenêtres qui se chevauchent sont fusionnées ;
    les bords tronqués reçoivent une ellipse. Aucune occurrence -> liste vide."""
    low = chunk.lower()
    spans: list[tuple[int, int]] = []
    for form in forms:
        f = form.lower().strip()
        if not f:
            continue
        start = 0
        while True:
            pos = low.find(f, start)
            if pos < 0:
                break
            spans.append((pos, pos + len(f)))
            start = pos + len(f)
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = []
    cs, ce = max(0, spans[0][0] - window), min(len(chunk), spans[0][1] + window)
    for s, e in spans[1:]:
        ws, we = max(0, s - window), min(len(chunk), e + window)
        if ws <= ce:                       # chevauche/adjacent -> fusionne
            ce = max(ce, we)
        else:
            merged.append((cs, ce))
            cs, ce = ws, we
    merged.append((cs, ce))
    out = []
    for s, e in merged:
        snippet = chunk[s:e].strip()
        if s > 0:
            snippet = "… " + snippet
        if e < len(chunk):
            snippet = snippet + " …"
        out.append(snippet)
    return out


def context_for_entity(
    conn: sqlite3.Connection, entity_id: int, chunks: list[str], *,
    window: int = 150, max_passages: int = 8, max_attributes: int = 40,
) -> dict:
    """Contexte d'une entité : attributs extraits, passages sources fenêtrés
    autour des mentions, résumés de moments. Passages et attributs sont plafonnés
    (`n_*_total` donne le total avant plafond, pour l'affichage « k / total »).
    `warnings` non vide si un chunk_index dépasse le nombre de chunks reconstitués
    (source/chunk_size incohérents)."""
    all_attrs = [
        {"name": name, "value": value}
        for name, value in conn.execute(
            "SELECT name, value FROM entity_attributes WHERE entity_id = ?", (entity_id,))
    ]
    forms = _entity_forms(conn, entity_id)
    chunk_indices = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT chunk_index FROM assertions "
            "WHERE entity_id = ? AND chunk_index IS NOT NULL ORDER BY chunk_index",
            (entity_id,))
    ]
    all_passages, warnings = [], []
    for i in chunk_indices:
        if 0 <= i < len(chunks):
            all_passages.extend(_windows_in_chunk(chunks[i], forms, window))
        else:
            warnings.append(f"chunk_index {i} hors bornes (source/chunk_size incohérents ?)")
    summaries = [
        row[0] for row in conn.execute(
            "SELECT m.summary FROM moments m JOIN appearances ap ON ap.moment_id = m.id "
            "WHERE ap.entity_id = ? AND m.summary <> ''", (entity_id,))
    ]
    return {
        "attributes": all_attrs[:max_attributes],
        "passages": all_passages[:max_passages],
        "summaries": summaries,
        "warnings": warnings,
        "n_attributes_total": len(all_attrs),
        "n_passages_total": len(all_passages),
    }
