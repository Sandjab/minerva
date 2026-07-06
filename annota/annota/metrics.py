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


def _link(size: int) -> float:
    return 1.0 if size == 1 else size * (size - 1) / 2.0


def _lea_directional(key: dict, resp: dict) -> float:
    """Score LEA directionnel (recall si key=gold,resp=pred ; precision si inversé).

    Chaque entité `key` pèse par sa taille (importance) et sa résolvabilité
    (fraction de ses liens internes retrouvés dans une même entité `resp`).
    Convention singleton (Moosavi & Strube 2016) : un singleton n'est résolu que
    s'il est aussi singleton côté `resp`."""
    key_clusters: dict = {}
    for m, c in key.items():
        key_clusters.setdefault(c, []).append(m)
    num = den = 0.0
    for members in key_clusters.values():
        size = len(members)
        importance = size
        den += importance
        if size == 1:
            (m,) = members
            resp_c = resp[m]
            is_resp_singleton = sum(1 for x in resp if resp[x] == resp_c) == 1
            resolution = 1.0 if is_resp_singleton else 0.0
        else:
            by_resp: dict = {}
            for m in members:
                by_resp[resp[m]] = by_resp.get(resp[m], 0) + 1
            found = sum(_link(cnt) for cnt in by_resp.values() if cnt >= 2)
            resolution = found / _link(size)
        num += importance * resolution
    return 0.0 if den == 0 else num / den


def lea(pred: dict, gold: dict) -> tuple[float, float, float]:
    """LEA (précision, rappel, F1) sur l'univers commun."""
    universe = _common_universe(pred, gold)
    if not universe:
        return (0.0, 0.0, 0.0)
    p = {m: pred[m] for m in universe}
    g = {m: gold[m] for m in universe}
    recall = _lea_directional(g, p)
    precision = _lea_directional(p, g)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)
