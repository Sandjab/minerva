"""Les formateurs CLI rendent l'historique temporel visible : une valeur qui
change entre deux moments doit se lire dans `show`, et `timeline` doit donner
l'ordre diégétique résolu."""

from minerva.cli import _format_entity, _format_timeline
from minerva.model import Assertion, Entity, KnowledgeGraph
from minerva.timeline import AVANT, Gap


def _graph():
    g = KnowledgeGraph()
    m1 = g.timeline.add_moment(0, 0, "enfance à Montfermeil")
    m2 = g.timeline.add_moment(1, 0, "dix ans après")
    g.timeline.add_constraint(m1.id, AVANT, m2.id, Gap(text="dix ans après", days=3650.0))
    g.add_entity(Entity(name="Cosette", type="personnage"))
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=1))
    g.add_assertion(Assertion(entity="Cosette", attribute="âge", value="18 ans", moment_id=2))
    g.timeline.resolve()
    return g


def test_show_entity_affiche_l_historique_des_valeurs():
    out = _format_entity(_graph(), "Cosette")
    assert "âge : 8 ans" in out          # vue first (comportement historique)
    assert "8 ans → 18 ans" in out       # historique diégétique


def test_show_entity_sans_historique_reste_sobre():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Javert", type="personnage", attributes={"métier": "inspecteur"}))
    out = _format_entity(g, "Javert")
    assert "→" not in out


def test_format_timeline_ordonne_et_montre_les_jours():
    out = _format_timeline(_graph())
    lines = out.splitlines()
    assert "enfance à Montfermeil" in lines[0]
    assert "dix ans après" in lines[1]
    assert "jour 3650" in lines[1]
    assert "2 moments" in lines[-1]


def test_format_timeline_graphe_sans_moments():
    out = _format_timeline(KnowledgeGraph())
    assert "aucun moment" in out.lower()
