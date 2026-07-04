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


def _tl(n_moments: int) -> Timeline:
    tl = Timeline()
    for i in range(n_moments):
        tl.add_moment(chunk_index=i, seq=0, summary=f"m{i + 1}")
    return tl


def test_resolve_sans_contrainte_suit_l_ordre_de_lecture():
    tl = _tl(3)
    tl.resolve()
    assert [m.resolved_order for m in tl.moments] == [0, 1, 2]
    assert tl.moment(1).resolved_days == 0.0  # origine : premier moment résolu


def test_resolve_flashback_passe_avant_sa_scene_de_lecture():
    # Lecture : M1 (présent), M2 (flashback), M3 (retour). Diégèse : M2 < M1 < M3.
    tl = _tl(3)
    tl.add_constraint(2, AVANT, 1)  # flashback : M2 avant M1
    tl.add_constraint(1, AVANT, 3)  # retour : M1 avant M3
    tl.resolve()
    orders = {m.id: m.resolved_order for m in tl.moments}
    assert orders[2] < orders[1] < orders[3]


def test_resolve_propage_les_ecarts_quantifies():
    tl = _tl(3)
    tl.add_constraint(1, AVANT, 2, Gap(text="vingt ans après", days=7300.0))
    tl.add_constraint(2, AVANT, 3, Gap(text="le lendemain", days=1.0))
    tl.resolve()
    assert tl.moment(1).resolved_days == 0.0
    assert tl.moment(2).resolved_days == 7300.0
    assert tl.moment(3).resolved_days == 7301.0


def test_resolve_ecart_non_quantifie_donne_days_none():
    tl = _tl(2)
    tl.add_constraint(1, AVANT, 2)  # avant, sans écart
    tl.resolve()
    assert tl.moment(1).resolved_days == 0.0
    assert tl.moment(2).resolved_days is None  # jamais de précision inventée


def test_resolve_flashback_quantifie_donne_des_jours_negatifs():
    tl = _tl(2)
    tl.add_constraint(2, AVANT, 1, Gap(text="dix ans plus tôt", days=3650.0))
    tl.resolve()
    # M2 précède M1 : l'origine (jour 0) est M2, M1 est à +3650.
    assert tl.moment(2).resolved_days == 0.0
    assert tl.moment(1).resolved_days == 3650.0


def test_resolve_simultane_regroupe_et_partage_la_coordonnee():
    tl = _tl(3)
    tl.add_constraint(1, AVANT, 2, Gap(days=10.0))
    tl.add_constraint(3, SIMULTANE, 2)  # fil parallèle
    tl.resolve()
    assert tl.moment(3).resolved_days == tl.moment(2).resolved_days == 10.0


def test_resolve_pendant_traite_comme_le_meme_groupe():
    tl = _tl(2)
    tl.add_constraint(2, PENDANT, 1)
    tl.resolve()
    assert tl.moment(2).resolved_days == tl.moment(1).resolved_days == 0.0


def test_resolve_cycle_signale_et_casse_sans_crash(caplog):
    tl = _tl(2)
    tl.add_constraint(1, AVANT, 2)
    tl.add_constraint(2, AVANT, 1)  # bruit LLM : cycle
    with caplog.at_level("WARNING"):
        tl.resolve()
    assert "cycle" in caplog.text.lower()
    # Repli : ordre de lecture.
    assert [m.resolved_order for m in tl.moments] == [0, 1]


def test_resolve_est_stable_et_recalculable():
    tl = _tl(4)
    tl.add_constraint(3, AVANT, 1)
    tl.resolve()
    first = [(m.resolved_order, m.resolved_days) for m in tl.moments]
    tl.resolve()
    assert [(m.resolved_order, m.resolved_days) for m in tl.moments] == first
