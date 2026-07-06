"""Vérité terrain d'annotation, persistée en SQLite pour écritures incrémentales."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS annotations (
    entity_id     INTEGER NOT NULL,
    kind          TEXT NOT NULL,            -- 'name' | 'alias'
    surface_form  TEXT NOT NULL,
    referent_id   TEXT,                     -- NULL = non décidé
    referent_type TEXT,
    discarded     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (entity_id, kind, surface_form)
);
"""


@dataclass
class Annotation:
    entity_id: int
    kind: str
    surface_form: str
    referent_id: str | None
    referent_type: str | None
    discarded: bool


class GoldStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @classmethod
    def create(cls, path, source_db: str) -> "GoldStore":
        conn = sqlite3.connect(str(path))
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('source_db', ?)", (source_db,))
        conn.commit()
        return cls(conn)

    @classmethod
    def open(cls, path) -> "GoldStore":
        return cls(sqlite3.connect(str(path)))

    def upsert(self, *, entity_id: int, kind: str, surface_form: str,
               referent_id: str | None = None, referent_type: str | None = None,
               discarded: bool = False) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO annotations "
            "(entity_id, kind, surface_form, referent_id, referent_type, discarded) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entity_id, kind, surface_form, referent_id, referent_type, int(discarded)),
        )
        self._conn.commit()

    def all(self) -> list[Annotation]:
        cur = self._conn.execute(
            "SELECT entity_id, kind, surface_form, referent_id, referent_type, discarded "
            "FROM annotations")
        return [Annotation(e, k, s, rid, rt, bool(d)) for e, k, s, rid, rt, d in cur]

    def gold_partition(self) -> dict:
        """{ (entity_id, kind, surface_form) -> referent_id } pour les surface
        forms décidées et non écartées."""
        return {
            (a.entity_id, a.kind, a.surface_form): a.referent_id
            for a in self.all()
            if a.referent_id is not None and not a.discarded
        }
