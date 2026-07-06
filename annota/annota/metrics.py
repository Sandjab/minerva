"""Métriques de clustering pour évaluer la fusion d'entités (B³, LEA, paires).

Partitions passées comme dict {clé -> id_de_cluster}. L'univers d'évaluation est
l'intersection des clés présentes dans les deux partitions (les surface forms
écartées/non annotées sont hors gold, donc exclues)."""
from __future__ import annotations

from itertools import combinations


def _common_universe(pred, gold):
    return [k for k in pred if k in gold]


def bcubed(pred: dict, gold: dict) -> tuple[float, float, float]:
    """B³ (précision, rappel, F1) sur l'univers commun.

    Pour chaque élément m : Pm = son cluster prédit, Gm = son cluster gold
    (tous deux restreints à l'univers). precision_m = |Pm∩Gm|/|Pm|,
    recall_m = |Pm∩Gm|/|Gm|. On moyenne sur les éléments."""
    universe = _common_universe(pred, gold)
    if not universe:
        return (0.0, 0.0, 0.0)
    pred_members: dict = {}
    gold_members: dict = {}
    for m in universe:
        pred_members.setdefault(pred[m], []).append(m)
        gold_members.setdefault(gold[m], []).append(m)
    prec_sum = rec_sum = 0.0
    for m in universe:
        pm = set(pred_members[pred[m]])
        gm = set(gold_members[gold[m]])
        inter = len(pm & gm)
        prec_sum += inter / len(pm)
        rec_sum += inter / len(gm)
    n = len(universe)
    precision = prec_sum / n
    recall = rec_sum / n
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def confusion_pairs(pred: dict, gold: dict) -> tuple[list, list]:
    """Retourne (over_merged, under_merged) : listes de paires (a, b) triées.

    over_merged : même cluster prédit, cluster gold différent (sur-fusion).
    under_merged : cluster prédit différent, même cluster gold (sous-fusion).
    Paires triées pour un affichage/déterminisme stable."""
    universe = _common_universe(pred, gold)
    over, under = [], []
    for a, b in combinations(sorted(universe), 2):
        same_pred = pred[a] == pred[b]
        same_gold = gold[a] == gold[b]
        if same_pred and not same_gold:
            over.append((a, b))
        elif same_gold and not same_pred:
            under.append((a, b))
    return (over, under)
