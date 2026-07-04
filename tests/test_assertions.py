"""Le journal de constats est la source de vérité : les dicts d'attributs
n'en sont qu'une vue « première extraction gagne », maintenue par un chemin
d'écriture unique (add_assertion). C'est ce qui permet à un attribut de
changer entre deux moments (Cosette : 8 ans -> 18 ans) sans rien casser."""

from minerva.model import Assertion, Entity, KnowledgeGraph, Relation
from minerva.timeline import AVANT, Gap


def _graph_with_moments(n: int) -> KnowledgeGraph:
    g = KnowledgeGraph()
    for i in range(n):
        g.timeline.add_moment(chunk_index=i, seq=0, summary=f"m{i + 1}")
    return g


def test_add_entity_avec_attributs_cree_des_assertions_moment_null():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Cosette", type="personnage", attributes={"âge": "8 ans"}))
    assert len(g.assertions) == 1
    a = g.assertions[0]
    assert (a.entity, a.attribute, a.value, a.moment_id) == ("Cosette", "âge", "8 ans", None)
    # La vue dict est maintenue à l'identique de l'ancien comportement.
    assert g.resolve("Cosette").attributes == {"âge": "8 ans"}


def test_meme_attribut_moments_differents_donne_deux_assertions():
    g = _graph_with_moments(2)
    g.add_entity(Entity(name="Cosette", type="personnage"))
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=1))
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="18 ans", moment_id=2))
    assert len(g.assertions) == 2
    # Vue « first » : la première valeur gagne, comme avant.
    assert g.resolve("Cosette").attributes == {"âge": "8 ans"}


def test_dedup_stricte_sujet_attribut_valeur_moment():
    g = _graph_with_moments(1)
    a = Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=1)
    assert g.add_assertion(a) is not None
    assert g.add_assertion(a.model_copy()) is None  # doublon exact écarté
    assert len(g.assertions) == 1


def test_valeurs_contradictoires_au_meme_moment_toutes_conservees():
    g = _graph_with_moments(1)
    g.add_assertion(Assertion(entity="X", attribute="âge", value="8 ans", moment_id=1))
    g.add_assertion(Assertion(entity="X", attribute="âge", value="9 ans", moment_id=1))
    assert len(g.assertions) == 2  # minerva ne tranche pas : matière de l'outil aval


def test_assertion_resout_les_alias_vers_le_nom_canonique():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Jean Valjean", type="personnage", aliases=["M. Madeleine"]))
    g.add_assertion(Assertion(entity="M. Madeleine", attribute="statut", value="maire"))
    assert g.assertions[-1].entity == "Jean Valjean"


def test_assertion_de_relation_observe_l_existence_a_un_moment():
    g = _graph_with_moments(1)
    g.add_assertion(
        Assertion(relation_name="héberge", relation_source="Thénardier",
                  relation_target="Cosette", moment_id=1)
    )
    assert len(g.relations) == 1  # la relation est créée au passage
    assert g.assertions[0].relation_name == "héberge"


def test_entity_state_final_prend_la_derniere_valeur_diegetique():
    g = _graph_with_moments(2)
    # Lecture : M1 puis M2, mais diégèse inversée (M2 est un flashback).
    g.timeline.add_constraint(2, AVANT, 1, Gap(text="dix ans plus tôt", days=3650.0))
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="18 ans", moment_id=1))
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=2))
    assert g.entity_state("first")["Cosette"]["âge"] == "18 ans"   # ordre de lecture
    assert g.entity_state("final")["Cosette"]["âge"] == "18 ans"   # M1 est diégétiquement après M2
    g2 = _graph_with_moments(2)
    g2.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=1))
    g2.add_assertion(Assertion(entity="Cosette", attribute="âge", value="18 ans", moment_id=2))
    assert g2.entity_state("final")["Cosette"]["âge"] == "18 ans"


def test_entity_state_final_une_valeur_datee_supplante_une_valeur_sans_moment():
    g = _graph_with_moments(1)
    g.add_entity(Entity(name="X", type="personnage", attributes={"âge": "8 ans"}))  # moment NULL
    g.add_assertion(Assertion(entity="X", attribute="âge", value="18 ans", moment_id=1))
    assert g.entity_state("final")["X"]["âge"] == "18 ans"
    assert g.entity_state("first")["X"]["âge"] == "8 ans"


def test_relation_state_first_et_final():
    g = _graph_with_moments(2)
    g.add_assertion(Assertion(relation_name="sert", relation_source="Cosette",
                              relation_target="Thénardier", attribute="statut",
                              value="servante", moment_id=1))
    g.add_assertion(Assertion(relation_name="sert", relation_source="Cosette",
                              relation_target="Thénardier", attribute="statut",
                              value="affranchie", moment_id=2))
    key = ("sert", "Cosette", "Thénardier")
    assert g.relation_state("first")[key]["statut"] == "servante"
    assert g.relation_state("final")[key]["statut"] == "affranchie"


def test_track_ordonne_les_moments_par_ordre_diegetique():
    g = _graph_with_moments(3)
    g.timeline.add_constraint(2, AVANT, 1)  # M2 = flashback avant M1
    g.timeline.add_constraint(1, AVANT, 3)
    g.add_entity(Entity(name="Cosette", type="personnage"))
    g.timeline.add_appearance(1, "Cosette")
    g.timeline.add_appearance(3, "Cosette")
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=2))
    track = g.track("Cosette")
    assert [m.id for m, _ in track] == [2, 1, 3]  # ordre diégétique, pas de lecture
    assert track[0][1][0].value == "8 ans"
    assert track[1][1] == []  # présence sans constat


def test_adopt_journal_remappe_les_sujets_vers_les_nouveaux_canoniques():
    src = _graph_with_moments(1)
    src.add_entity(Entity(name="évêque Myriel", type="personnage"))
    src.timeline.add_appearance(1, "évêque Myriel")
    src.add_assertion(Assertion(entity="évêque Myriel", attribute="ville", value="Digne", moment_id=1))
    dst = KnowledgeGraph()
    dst.add_entity(Entity(name="Monseigneur Bienvenu", type="personnage",
                          aliases=["évêque Myriel"]))
    dst.adopt_journal(src)
    assert dst.assertions[0].entity == "Monseigneur Bienvenu"
    assert dst.timeline.appearances == {1: {"Monseigneur Bienvenu"}}
    assert dst.resolve("Monseigneur Bienvenu").attributes == {"ville": "Digne"}


def test_to_dict_inclut_le_journal():
    g = _graph_with_moments(1)
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=1))
    d = g.to_dict()
    assert {"entities", "relations", "moments", "constraints", "appearances", "assertions"} <= set(d)
    assert d["assertions"][0]["value"] == "8 ans"
