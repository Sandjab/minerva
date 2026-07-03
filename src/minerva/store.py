"""Persistance SQLite du graphe (schéma EAV)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .model import Entity, KnowledgeGraph, Relation

_SCHEMA = """
CREATE TABLE entities (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL
);
CREATE TABLE entity_aliases (
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    alias     TEXT NOT NULL,
    UNIQUE (entity_id, alias)
);
CREATE TABLE entity_attributes (
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    name      TEXT NOT NULL,
    value     TEXT NOT NULL,
    UNIQUE (entity_id, name)
);
CREATE TABLE relations (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    source_id INTEGER NOT NULL REFERENCES entities(id),
    target_id INTEGER NOT NULL REFERENCES entities(id),
    UNIQUE (name, source_id, target_id)
);
CREATE TABLE relation_attributes (
    relation_id INTEGER NOT NULL REFERENCES relations(id),
    name        TEXT NOT NULL,
    value       TEXT NOT NULL,
    UNIQUE (relation_id, name)
);
"""


def save(graph: KnowledgeGraph, path: str | Path) -> None:
    """Écrit le graphe dans une base SQLite (écrase le fichier existant)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        entity_ids: dict[str, int] = {}
        for entity in graph.entities:
            cur = conn.execute(
                "INSERT INTO entities (name, type) VALUES (?, ?)",
                (entity.name, entity.type),
            )
            eid = int(cur.lastrowid or 0)
            entity_ids[entity.name] = eid
            conn.executemany(
                "INSERT INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
                [(eid, a) for a in entity.aliases],
            )
            conn.executemany(
                "INSERT INTO entity_attributes (entity_id, name, value) VALUES (?, ?, ?)",
                [(eid, k, v) for k, v in entity.attributes.items()],
            )
        for relation in graph.relations:
            cur = conn.execute(
                "INSERT INTO relations (name, source_id, target_id) VALUES (?, ?, ?)",
                (relation.name, entity_ids[relation.source], entity_ids[relation.target]),
            )
            conn.executemany(
                "INSERT INTO relation_attributes (relation_id, name, value) VALUES (?, ?, ?)",
                [(cur.lastrowid, k, v) for k, v in relation.attributes.items()],
            )
        conn.commit()
    finally:
        conn.close()


def load(path: str | Path) -> KnowledgeGraph:
    """Reconstruit un KnowledgeGraph depuis une base SQLite."""
    conn = sqlite3.connect(path)
    try:
        graph = KnowledgeGraph()
        names: dict[int, str] = {}
        for eid, name, etype in conn.execute("SELECT id, name, type FROM entities"):
            names[eid] = name
            aliases = [
                row[0]
                for row in conn.execute(
                    "SELECT alias FROM entity_aliases WHERE entity_id = ?", (eid,)
                )
            ]
            attributes = dict(
                conn.execute(
                    "SELECT name, value FROM entity_attributes WHERE entity_id = ?", (eid,)
                )
            )
            graph.add_entity(
                Entity(name=name, type=etype, aliases=aliases, attributes=attributes)
            )
        for rid, name, source_id, target_id in conn.execute(
            "SELECT id, name, source_id, target_id FROM relations"
        ):
            attributes = dict(
                conn.execute(
                    "SELECT name, value FROM relation_attributes WHERE relation_id = ?",
                    (rid,),
                )
            )
            graph.add_relation(
                Relation(
                    name=name,
                    source=names[source_id],
                    target=names[target_id],
                    attributes=attributes,
                )
            )
        return graph
    finally:
        conn.close()
