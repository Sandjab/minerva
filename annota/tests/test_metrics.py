import math

from annota.metrics import bcubed, confusion_pairs


def _almost(a, b):
    return math.isclose(a, b, abs_tol=1e-9)


def test_identical_partitions_are_perfect():
    # Intention : aucune erreur → B³ parfait, aucune paire fautive.
    pred = {"a": "1", "b": "1", "c": "2"}
    gold = {"a": "X", "b": "X", "c": "Y"}   # mêmes regroupements, ids libres
    p, r, f = bcubed(pred, gold)
    assert (p, r, f) == (1.0, 1.0, 1.0)
    over, under = confusion_pairs(pred, gold)
    assert over == [] and under == []


def test_total_over_merge_tanks_precision():
    # Intention : canon_alias réunit 2 référents distincts → précision chute,
    # rappel intact, et les paires sur-fusionnées sont listées.
    pred = {"a": "1", "b": "1", "c": "1", "d": "1"}   # tout ensemble
    gold = {"a": "X", "b": "X", "c": "Y", "d": "Y"}   # deux vrais référents
    p, r, f = bcubed(pred, gold)
    assert _almost(p, 0.5) and _almost(r, 1.0)
    over, under = confusion_pairs(pred, gold)
    assert sorted(over) == [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
    assert under == []


def test_total_under_merge_tanks_recall():
    # Intention : canon_alias laisse un même référent éclaté → rappel chute.
    pred = {"a": "1", "b": "2", "c": "3", "d": "4"}   # tout séparé
    gold = {"a": "X", "b": "X", "c": "X", "d": "X"}   # un seul référent
    p, r, f = bcubed(pred, gold)
    assert _almost(p, 1.0) and _almost(r, 0.25)
    over, under = confusion_pairs(pred, gold)
    assert over == [] and len(under) == 6  # C(4,2)


def test_discards_excluded_from_universe():
    # Intention : une surface form absente du gold (discard) ne compte pas.
    pred = {"a": "1", "b": "1", "noise": "1"}
    gold = {"a": "X", "b": "X"}              # 'noise' non annoté / écarté
    p, r, f = bcubed(pred, gold)
    assert (p, r, f) == (1.0, 1.0, 1.0)      # comme si 'noise' n'existait pas
