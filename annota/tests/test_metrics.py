import math

from annota.metrics import bcubed, confusion_pairs, lea


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


def test_lea_identical_is_one():
    pred = {"a": "1", "b": "1", "c": "2"}
    gold = {"a": "X", "b": "X", "c": "Y"}
    p, r, f = lea(pred, gold)
    assert _almost(p, 1.0) and _almost(r, 1.0) and _almost(f, 1.0)


def test_lea_over_merge_penalizes_precision():
    # pred fusionne {a,b,c} ; gold = {a,b} + {c}
    pred = {"a": "1", "b": "1", "c": "1"}
    gold = {"a": "X", "b": "X", "c": "Y"}
    # recall : gold {a,b} (link=1) retrouvé dans pred → resolution=1 ; gold {c}
    #   singleton mais NON singleton dans pred → resolution=0 ; (2·1+1·0)/3 = 2/3
    # precision : pred {a,b,c} (link=3) ; seul lien (a,b) dans un même gold → 1/3 ;
    #   importance 3 → (3·(1/3))/3 = 1/3
    p, r, f = lea(pred, gold)
    assert _almost(r, 2 / 3) and _almost(p, 1 / 3)
