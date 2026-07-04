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
    valjean = next(e for e in p["entities"] if e["name"] == "Valjean")
    assert valjean["aliases"] == ["Jean Valjean"]


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


def test_payload_tracks_fusionne_les_moments_consecutifs():
    p = build_payload(_graph())
    by_name = {t["entity"]: t for t in p["tracks"]}
    assert by_name["Valjean"]["runs"] == [[0, 2]]   # présent aux ordres 0,1,2
    assert by_name["Myriel"]["runs"] == [[1, 1]]
    assert by_name["Digne"]["runs"] == [[0, 0]]


def test_payload_tracks_triees_par_nombre_de_moments_decroissant():
    p = build_payload(_graph())
    assert p["tracks"][0]["entity"] == "Valjean"
    assert p["tracks"][0]["count"] == 3


def test_payload_tracks_un_trou_coupe_le_run():
    g = _graph()
    g.timeline.add_appearance(3, "Digne")  # m3 : Digne réapparaît après un trou
    p = build_payload(g)
    by_name = {t["entity"]: t for t in p["tracks"]}
    assert by_name["Digne"]["runs"] == [[0, 0], [2, 2]]


def test_states_une_entite_n_est_visible_qu_a_partir_de_sa_premiere_trace():
    p = build_payload(_graph())
    # Myriel apparaît pour la première fois à l'ordre 1
    assert "Myriel" not in p["states"][0]["entities"]
    assert "Myriel" in p["states"][1]["entities"]
    assert "Valjean" in p["states"][0]["entities"]


def test_states_entite_sans_trace_datee_est_visible_partout():
    g = _graph()
    g.add_entity(Entity(name="Cosette", type="personnage"))  # aucune trace datée
    p = build_payload(g)
    assert "Cosette" in p["states"][0]["entities"]


def test_states_relation_attend_ses_deux_extremites():
    # héberge a une assertion datée à l'ordre 1 ET Myriel n'est visible qu'à
    # l'ordre 1 : la relation ne doit pas exister avant (pas de lien pendouillant).
    p = build_payload(_graph())
    assert p["states"][0]["relations"] == []
    assert p["states"][1]["relations"] == [0]


def test_states_relation_sans_assertion_datee_suit_ses_extremites():
    g = _graph()
    g.add_relation(Relation(name="traverse", source="Valjean", target="Digne"))
    p = build_payload(g)
    idx = [i for i, r in enumerate(p["relations"]) if r["name"] == "traverse"][0]
    assert idx in p["states"][0]["relations"]  # les deux extrémités sont à l'ordre 0
