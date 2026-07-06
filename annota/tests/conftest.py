import sqlite3

import pytest

# Schéma minimal minerva utilisé par reader (contrat d'interface).
_MINERVA_SCHEMA = """
CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, type TEXT NOT NULL);
CREATE TABLE entity_aliases (entity_id INTEGER NOT NULL, alias TEXT NOT NULL, UNIQUE(entity_id, alias));
CREATE TABLE moments (id INTEGER PRIMARY KEY, chunk_index INTEGER NOT NULL, seq INTEGER NOT NULL, summary TEXT NOT NULL DEFAULT '');
CREATE TABLE appearances (moment_id INTEGER NOT NULL, entity_id INTEGER NOT NULL, UNIQUE(moment_id, entity_id));
CREATE TABLE assertions (id INTEGER PRIMARY KEY, entity_id INTEGER, relation_id INTEGER,
    attribute TEXT NOT NULL DEFAULT '', value TEXT NOT NULL DEFAULT '', moment_id INTEGER, chunk_index INTEGER);
CREATE VIEW entity_attributes AS
  SELECT a.entity_id, a.attribute AS name, a.value FROM assertions a
  WHERE a.entity_id IS NOT NULL AND a.attribute <> ''
    AND a.id = (SELECT MIN(b.id) FROM assertions b WHERE b.entity_id=a.entity_id AND b.attribute=a.attribute);
"""


@pytest.fixture
def minerva_db(tmp_path):
    """Base minerva synthétique : 2 entités (1 avec alias + attributs), assertions
    ancrées à des chunks, 1 moment résumé. Source = 2 chunks."""
    path = tmp_path / "mini.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_MINERVA_SCHEMA)
    conn.execute("INSERT INTO entities VALUES (1, 'Élise', 'personnage')")
    conn.execute("INSERT INTO entities VALUES (2, 'Anouck', 'inconnu')")
    conn.execute("INSERT INTO entity_aliases VALUES (1, 'Élise Blanchard')")
    conn.execute("INSERT INTO moments VALUES (10, 0, 0, 'Élise arrive.')")
    conn.execute("INSERT INTO appearances VALUES (10, 1)")
    # attribut d'Élise, ancré au chunk 0 ; mention d'Anouck au chunk 1
    conn.execute("INSERT INTO assertions VALUES (100, 1, NULL, 'rôle', 'protagoniste', 10, 0)")
    conn.execute("INSERT INTO assertions VALUES (101, 2, NULL, '', '', NULL, 1)")
    conn.commit()
    # source dont split_text(_, 8000) redonne exactement 2 chunks (2 paragraphes)
    source = tmp_path / "src.md"
    source.write_text("Chunk zéro : Élise arrive.\n\nChunk un : Anouck observe.", encoding="utf-8")
    return {"db": path, "source": source}
