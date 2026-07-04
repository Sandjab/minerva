"""Le temps diégétique est un graphe de contraintes entre moments narratifs,
jamais un axe absolu : on ne stocke que ce que le texte affirme."""

import pytest

from minerva.timeline import (
    AVANT, PENDANT, SIMULTANE, Gap, Moment, TemporalConstraint, Timeline, gap_to_days,
)


def test_gap_to_days_unites_francaises():
    assert gap_to_days(20, "années") == 20 * 365
    assert gap_to_days(20, "ans") == 20 * 365
    assert gap_to_days(3, "jours") == 3
    assert gap_to_days(2, "semaines") == 14
    assert gap_to_days(6, "mois") == 180
    assert gap_to_days(12, "heures") == 0.5


def test_gap_to_days_inconnu_ou_vide():
    assert gap_to_days(None, "ans") is None
    assert gap_to_days(0, "ans") is None
    assert gap_to_days(5, "lustres") is None  # unité inconnue : pas d'invention
    assert gap_to_days(5, "") is None


def test_add_moment_assigne_des_ids_globaux_croissants():
    tl = Timeline()
    m1 = tl.add_moment(chunk_index=0, seq=0, summary="ouverture")
    m2 = tl.add_moment(chunk_index=0, seq=1, summary="flashback")
    m3 = tl.add_moment(chunk_index=1, seq=0, summary="suite")
    assert (m1.id, m2.id, m3.id) == (1, 2, 3)
    assert tl.moment(2) is m2
    assert [m.id for m in tl.moments] == [1, 2, 3]


def test_add_constraint_valide_les_ids_et_la_relation():
    tl = Timeline()
    m1 = tl.add_moment(0, 0, "a")
    m2 = tl.add_moment(0, 1, "b")
    tl.add_constraint(m1.id, AVANT, m2.id, Gap(text="le lendemain", days=1.0))
    assert len(tl.constraints) == 1
    with pytest.raises(ValueError):
        tl.add_constraint(m1.id, "après", m2.id)  # relation inconnue
    with pytest.raises(ValueError):
        tl.add_constraint(m1.id, AVANT, 99)  # moment inexistant


def test_appearances_dedupliquees_par_moment():
    tl = Timeline()
    m = tl.add_moment(0, 0, "scène")
    tl.add_appearance(m.id, "Cosette")
    tl.add_appearance(m.id, "Cosette")
    assert tl.appearances == {m.id: {"Cosette"}}


def test_recent_donne_les_derniers_moments_en_ordre_de_lecture():
    tl = Timeline()
    for i in range(5):
        tl.add_moment(i, 0, f"scène {i}")
    assert [m.summary for m in tl.recent(2)] == ["scène 3", "scène 4"]
