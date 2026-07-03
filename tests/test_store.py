"""L'aller-retour SQLite doit être sans perte : le graphe rechargé est le
support de toutes les requêtes ultérieures (show, export, SQL direct)."""

from minerva import store
from minerva.model import Entity, KnowledgeGraph, Relation


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
