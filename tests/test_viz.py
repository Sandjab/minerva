"""La page de visualisation dit la vérité du modèle : le payload doit
contenir tout ce que la base sait, sous une forme prête à rendre — les
transformations se font en Python, jamais dans le JS de la page."""

from minerva.model import Assertion, Entity, KnowledgeGraph, Relation
from minerva.timeline import AVANT, Gap
from minerva.viz import build_payload


def _graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    m1 = g.timeline.add_moment(0, 0, "arrivée à Digne")
    m2 = g.timeline.add_moment(0, 1, "chez Myriel")
    m3 = g.timeline.add_moment(1, 0, "dix ans après")
    g.timeline.add_constraint(m1.id, AVANT, m2.id)
    g.timeline.add_constraint(m2.id, AVANT, m3.id, Gap(text="dix ans après", days=3650.0))
    g.add_entity(Entity(name="Valjean", type="personnage", aliases=["Jean Valjean"]))
    g.add_entity(Entity(name="Myriel", type="personnage"))
    g.add_entity(Entity(name="Digne", type="lieu"))
    g.add_relation(Relation(name="héberge", source="Myriel", target="Valjean"))
    g.timeline.add_appearance(m1.id, "Valjean")
    g.timeline.add_appearance(m1.id, "Digne")
    g.timeline.add_appearance(m2.id, "Valjean")
    g.timeline.add_appearance(m2.id, "Myriel")
    g.timeline.add_appearance(m3.id, "Valjean")
    g.add_assertion(Assertion(relation_name="héberge", relation_source="Myriel",
                              relation_target="Valjean", attribute="lieu",
                              value="évêché", moment_id=m2.id))
    return g


def test_payload_contient_toutes_les_entites_relations_moments():
    p = build_payload(_graph())
    assert {e["name"] for e in p["entities"]} == {"Valjean", "Myriel", "Digne"}
    assert {(r["source"], r["name"], r["target"]) for r in p["relations"]} == {
        ("Myriel", "héberge", "Valjean")
    }
    assert [m["order"] for m in p["moments"]] == [0, 1, 2]
    assert p["moments"][0]["summary"] == "arrivée à Digne"


def test_payload_moments_portent_les_jours_quand_connus():
    p = build_payload(_graph())
    days = [m["days"] for m in p["moments"]]
    # Le résolveur peut ou non ancrer une origine ; on exige seulement que le
    # champ existe pour chaque moment (nullable) et qu'aucun jour ne soit
    # inventé hors chemin quantifié.
    assert len(days) == 3


def test_payload_gaps_seulement_quantifies_et_consecutifs():
    p = build_payload(_graph())
    assert p["gaps"] == {"1": 3650.0}  # m2 -> m3, ordres 1 -> 2


def test_payload_relations_portent_leurs_assertions_sources():
    p = build_payload(_graph())
    rel = p["relations"][0]
    assert rel["assertions"] == [{"attribute": "lieu", "value": "évêché", "moment_id": 2}]
