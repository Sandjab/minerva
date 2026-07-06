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


def context_for_entity(conn: sqlite3.Connection, entity_id: int, chunks: list[str]) -> dict:
    """Contexte d'une entité : attributs extraits, passages sources (via
    chunk_index), résumés de moments. `warnings` non vide si un chunk_index
    dépasse le nombre de chunks reconstitués (source/chunk_size incohérents)."""
    attributes = [
        {"name": name, "value": value}
        for name, value in conn.execute(
            "SELECT name, value FROM entity_attributes WHERE entity_id = ?", (entity_id,))
    ]
    chunk_indices = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT chunk_index FROM assertions "
            "WHERE entity_id = ? AND chunk_index IS NOT NULL ORDER BY chunk_index",
            (entity_id,))
    ]
    passages, warnings = [], []
    for i in chunk_indices:
        if 0 <= i < len(chunks):
            passages.append(chunks[i])
        else:
            warnings.append(f"chunk_index {i} hors bornes (source/chunk_size incohérents ?)")
    summaries = [
        row[0] for row in conn.execute(
            "SELECT m.summary FROM moments m JOIN appearances ap ON ap.moment_id = m.id "
            "WHERE ap.entity_id = ? AND m.summary <> ''", (entity_id,))
    ]
    return {"attributes": attributes, "passages": passages,
            "summaries": summaries, "warnings": warnings}
