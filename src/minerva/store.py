"""Persistance SQLite du graphe : identités + journal de constats.

Le journal (assertions, moments, contraintes, présences) est la source de
vérité ; les anciennes tables d'attributs deviennent des VUES SQL
« première valeur » pour garder les requêtes existantes fonctionnelles.
Les bases à l'ancien schéma restent chargeables en mode dégradé (attributs
convertis en constats non datés)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .model import Assertion, Entity, KnowledgeGraph, Relation
from .timeline import Gap, Moment

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
CREATE TABLE relations (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    source_id INTEGER NOT NULL REFERENCES entities(id),
    target_id INTEGER NOT NULL REFERENCES entities(id),
    UNIQUE (name, source_id, target_id)
);
CREATE TABLE moments (
    id             INTEGER PRIMARY KEY,
    chunk_index    INTEGER NOT NULL,
    seq            INTEGER NOT NULL,
    summary        TEXT NOT NULL DEFAULT '',
    resolved_order INTEGER,  -- dérivé (résolveur), recalculable
    resolved_days  REAL      -- dérivé (résolveur), recalculable
);
CREATE TABLE moment_constraints (
    id        INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES moments(id),
    relation  TEXT NOT NULL,
    target_id INTEGER NOT NULL REFERENCES moments(id),
    gap_text  TEXT NOT NULL DEFAULT '',
    gap_days  REAL
);
CREATE TABLE appearances (
    moment_id INTEGER NOT NULL REFERENCES moments(id),
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    UNIQUE (moment_id, entity_id)
);
CREATE TABLE assertions (
    id          INTEGER PRIMARY KEY,
    entity_id   INTEGER REFERENCES entities(id),
    relation_id INTEGER REFERENCES relations(id),
    attribute   TEXT NOT NULL DEFAULT '',
    value       TEXT NOT NULL DEFAULT '',
    moment_id   INTEGER REFERENCES moments(id),
    chunk_index INTEGER,
    CHECK ((entity_id IS NULL) <> (relation_id IS NULL))
);
-- Vues de compatibilité : première valeur par (sujet, attribut), comme
-- l'ancienne sémantique « première extraction gagne ».
CREATE VIEW entity_attributes AS
SELECT a.entity_id, a.attribute AS name, a.value
FROM assertions a
WHERE a.entity_id IS NOT NULL AND a.attribute <> ''
  AND a.id = (SELECT MIN(b.id) FROM assertions b
              WHERE b.entity_id = a.entity_id AND b.attribute = a.attribute);
CREATE VIEW relation_attributes AS
SELECT a.relation_id, a.attribute AS name, a.value
FROM assertions a
WHERE a.relation_id IS NOT NULL AND a.attribute <> ''
  AND a.id = (SELECT MIN(b.id) FROM assertions b
              WHERE b.relation_id = a.relation_id AND b.attribute = a.attribute);
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
        relation_ids: dict[tuple[str, str, str], int] = {}
        for relation in graph.relations:
            cur = conn.execute(
                "INSERT INTO relations (name, source_id, target_id) VALUES (?, ?, ?)",
                (relation.name, entity_ids[relation.source], entity_ids[relation.target]),
            )
            relation_ids[(relation.name, relation.source, relation.target)] = int(
                cur.lastrowid or 0
            )
        for m in graph.timeline.moments:
            conn.execute(
                "INSERT INTO moments (id, chunk_index, seq, summary, resolved_order,"
                " resolved_days) VALUES (?, ?, ?, ?, ?, ?)",
                (m.id, m.chunk_index, m.seq, m.summary, m.resolved_order, m.resolved_days),
            )
        for c in graph.timeline.constraints:
            conn.execute(
                "INSERT INTO moment_constraints (source_id, relation, target_id,"
                " gap_text, gap_days) VALUES (?, ?, ?, ?, ?)",
                (c.source_id, c.relation, c.target_id, c.gap.text, c.gap.days),
            )
        for moment_id, names in graph.timeline.appearances.items():
            conn.executemany(
                "INSERT INTO appearances (moment_id, entity_id) VALUES (?, ?)",
                [(moment_id, entity_ids[n]) for n in sorted(names) if n in entity_ids],
            )
        for a in graph.assertions:
            conn.execute(
                "INSERT INTO assertions (entity_id, relation_id, attribute, value,"
                " moment_id, chunk_index) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entity_ids.get(a.entity) if a.entity else None,
                    relation_ids.get((a.relation_name, a.relation_source, a.relation_target))
                    if a.relation_name else None,
                    a.attribute, a.value, a.moment_id, a.chunk_index,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load(path: str | Path) -> KnowledgeGraph:
    """Reconstruit un KnowledgeGraph. Détecte l'ancien schéma (pas de table
    assertions) et le charge en mode dégradé : les attributs deviennent des
    constats non datés via add_entity/add_relation."""
    conn = sqlite3.connect(path)
    try:
        has_journal = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='assertions'"
        ).fetchone()
        return _load_journal(conn) if has_journal else _load_legacy(conn)
    finally:
        conn.close()


def _load_identities(conn: sqlite3.Connection, graph: KnowledgeGraph) -> tuple[dict, dict]:
    entity_names: dict[int, str] = {}
    for eid, name, etype in conn.execute("SELECT id, name, type FROM entities ORDER BY id"):
        entity_names[eid] = name
        aliases = [
            row[0]
            for row in conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id = ?", (eid,)
            )
        ]
        graph.add_entity(Entity(name=name, type=etype, aliases=aliases))
    relation_keys: dict[int, tuple[str, str, str]] = {}
    for rid, name, source_id, target_id in conn.execute(
        "SELECT id, name, source_id, target_id FROM relations ORDER BY id"
    ):
        relation = graph.add_relation(
            Relation(name=name, source=entity_names[source_id], target=entity_names[target_id])
        )
        relation_keys[rid] = (relation.name, relation.source, relation.target)
    return entity_names, relation_keys


def _load_journal(conn: sqlite3.Connection) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    entity_names, relation_keys = _load_identities(conn, graph)
    for mid, chunk_index, seq, summary, r_order, r_days in conn.execute(
        "SELECT id, chunk_index, seq, summary, resolved_order, resolved_days"
        " FROM moments ORDER BY id"
    ):
        graph.timeline.load_moment(
            Moment(id=mid, chunk_index=chunk_index, seq=seq, summary=summary,
                   resolved_order=r_order, resolved_days=r_days)
        )
    for source_id, relation, target_id, gap_text, gap_days in conn.execute(
        "SELECT source_id, relation, target_id, gap_text, gap_days"
        " FROM moment_constraints ORDER BY id"
    ):
        graph.timeline.add_constraint(
            source_id, relation, target_id, Gap(text=gap_text, days=gap_days)
        )
    for moment_id, entity_id in conn.execute(
        "SELECT moment_id, entity_id FROM appearances"
    ):
        graph.timeline.add_appearance(moment_id, entity_names[entity_id])
    for entity_id, relation_id, attribute, value, moment_id, chunk_index in conn.execute(
        "SELECT entity_id, relation_id, attribute, value, moment_id, chunk_index"
        " FROM assertions ORDER BY id"
    ):
        if entity_id is not None:
            assertion = Assertion(entity=entity_names[entity_id], attribute=attribute,
                                  value=value, moment_id=moment_id, chunk_index=chunk_index)
        else:
            name, source, target = relation_keys[relation_id]
            assertion = Assertion(relation_name=name, relation_source=source,
                                  relation_target=target, attribute=attribute,
                                  value=value, moment_id=moment_id, chunk_index=chunk_index)
        graph.add_assertion(assertion)
    return graph


def _load_legacy(conn: sqlite3.Connection) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    names: dict[int, str] = {}
    for eid, name, etype in conn.execute("SELECT id, name, type FROM entities ORDER BY id"):
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
        graph.add_entity(Entity(name=name, type=etype, aliases=aliases, attributes=attributes))
    for rid, name, source_id, target_id in conn.execute(
        "SELECT id, name, source_id, target_id FROM relations ORDER BY id"
    ):
        attributes = dict(
            conn.execute(
                "SELECT name, value FROM relation_attributes WHERE relation_id = ?", (rid,)
            )
        )
        graph.add_relation(
            Relation(name=name, source=names[source_id], target=names[target_id],
                     attributes=attributes)
        )
    return graph
