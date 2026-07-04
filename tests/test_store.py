"""L'aller-retour SQLite doit être sans perte : le graphe rechargé est le
support de toutes les requêtes ultérieures (show, export, SQL direct)."""

import sqlite3

from minerva import store
from minerva.model import Assertion, Entity, KnowledgeGraph, Relation
from minerva.timeline import AVANT, Gap


def _sample_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_entity(
        Entity(
            name="Jean Valjean",
            type="personnage",
            aliases=["M. Madeleine"],
            attributes={"profession": "forçat", "âge": "25 ans"},
        )
    )
    g.add_entity(Entity(name="Toulon", type="lieu"))
    g.add_relation(
        Relation(
            name="emprisonné à",
            source="Jean Valjean",
            target="Toulon",
            attributes={"durée": "19 ans"},
        )
    )
    return g


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "graphe.db"
    store.save(_sample_graph(), path)
    loaded = store.load(path)

    assert loaded.to_dict() == _sample_graph().to_dict()


def test_save_overwrites_existing_file(tmp_path):
    path = tmp_path / "graphe.db"
    store.save(_sample_graph(), path)
    g2 = KnowledgeGraph()
    g2.add_entity(Entity(name="Cosette", type="personnage"))
    store.save(g2, path)

    loaded = store.load(path)
    assert [e.name for e in loaded.entities] == ["Cosette"]


def test_round_trip_du_journal(tmp_path):
    g = KnowledgeGraph()
    m1 = g.timeline.add_moment(0, 0, "enfance")
    m2 = g.timeline.add_moment(1, 0, "vingt ans après")
    g.timeline.add_constraint(m1.id, AVANT, m2.id, Gap(text="vingt ans après", days=7300.0))
    g.add_entity(Entity(name="Cosette", type="personnage", aliases=["l'enfant"]))
    g.timeline.add_appearance(m1.id, "Cosette")
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans",
                              moment_id=m1.id, chunk_index=0))
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="28 ans",
                              moment_id=m2.id, chunk_index=1))
    g.add_assertion(Assertion(relation_name="protège", relation_source="Jean Valjean",
                              relation_target="Cosette", moment_id=m2.id, chunk_index=1))
    g.timeline.resolve()

    path = tmp_path / "g.db"
    store.save(g, path)
    loaded = store.load(path)

    assert [m.model_dump() for m in loaded.timeline.moments] == [
        m.model_dump() for m in g.timeline.moments
    ]
    assert len(loaded.timeline.constraints) == 1
    assert loaded.timeline.constraints[0].gap.days == 7300.0
    assert loaded.timeline.appearances == g.timeline.appearances
    assert [a.model_dump() for a in loaded.assertions] == [
        a.model_dump() for a in g.assertions
    ]
    # Vue dict reconstruite en rejouant le journal (first-wins).
    assert loaded.resolve("Cosette").attributes == {"âge": "8 ans"}
    assert loaded.resolve("Cosette").aliases == ["l'enfant"]


def test_vues_sql_premiere_valeur(tmp_path):
    g = KnowledgeGraph()
    g.timeline.add_moment(0, 0, "m1")
    g.timeline.add_moment(1, 0, "m2")
    g.add_assertion(Assertion(entity="X", attribute="âge", value="8", moment_id=1))
    g.add_assertion(Assertion(entity="X", attribute="âge", value="18", moment_id=2))
    path = tmp_path / "g.db"
    store.save(g, path)
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT e.name, a.name, a.value FROM entity_attributes a "
        "JOIN entities e ON e.id = a.entity_id"
    ).fetchall()
    conn.close()
    assert rows == [("X", "âge", "8")]  # la vue SQL reproduit first-wins


def test_chargement_legacy_en_mode_degrade(tmp_path):
    # Base à l'ancien schéma : attributs en tables, pas de journal.
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                               type TEXT NOT NULL);
        CREATE TABLE entity_aliases (entity_id INTEGER NOT NULL, alias TEXT NOT NULL,
                                     UNIQUE (entity_id, alias));
        CREATE TABLE entity_attributes (entity_id INTEGER NOT NULL, name TEXT NOT NULL,
                                        value TEXT NOT NULL, UNIQUE (entity_id, name));
        CREATE TABLE relations (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                                source_id INTEGER NOT NULL, target_id INTEGER NOT NULL,
                                UNIQUE (name, source_id, target_id));
        CREATE TABLE relation_attributes (relation_id INTEGER NOT NULL, name TEXT NOT NULL,
                                          value TEXT NOT NULL, UNIQUE (relation_id, name));
        INSERT INTO entities (id, name, type) VALUES (1, 'Cosette', 'personnage');
        INSERT INTO entity_attributes VALUES (1, 'âge', '8 ans');
    """)
    conn.commit()
    conn.close()

    loaded = store.load(path)
    assert loaded.resolve("Cosette").attributes == {"âge": "8 ans"}
    [a] = loaded.assertions
    assert (a.attribute, a.moment_id) == ("âge", None)  # constat non daté
    assert loaded.timeline.moments == []
