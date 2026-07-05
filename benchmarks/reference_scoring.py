"""Scoring exhaustif : précision/rappel d'un graphe contre une référence déclarative.

Référence : docs/specs/2026-07-05-minerva-reference-design.md. Module de
benchmarks/, chargé par importlib depuis les benchs, rescore.py et les tests
(motif rescore.py). Déterministe : aucun juge LLM.
"""

import json
from pathlib import Path

from minerva.model import Entity, KnowledgeGraph, normalize, strip_title

VALID_LEVELS = {"core", "optional"}


def _mention_keys(mention: str) -> set[str]:
    """Clés d'appariement d'une mention : forme normalisée + forme sans
    article/titre (même repli que la résolution du graphe)."""
    key = normalize(mention)
    keys = {key}
    stripped = strip_title(key)
    if stripped:
        keys.add(stripped)
    return keys


class Reference:
    """Référence exhaustive : entités à deux niveaux, paires de relations
    non orientées, fusions d'alias exigées."""

    def __init__(self, data: dict):
        self.data = data
        self.entries = data["entities"]
        self.order = {e["name"]: i for i, e in enumerate(self.entries)}
        self.level = {e["name"]: e["level"] for e in self.entries}
        self.variant_index: dict[str, str] = {}
        self.variant_collisions: list[str] = []
        for entry in self.entries:
            for variant in entry["variants"]:
                for key in _mention_keys(variant):
                    owner = self.variant_index.setdefault(key, entry["name"])
                    if owner != entry["name"]:
                        self.variant_collisions.append(
                            f"variant « {variant} » de « {entry['name']} » "
                            f"déjà revendiqué par « {owner} »"
                        )
        self.core_pairs = {
            frozenset(r["pair"]) for r in data["relations"] if r["level"] == "core"
        }
        self.optional_pairs = {
            frozenset(r["pair"]) for r in data["relations"] if r["level"] == "optional"
        }
        self.required_merges = [
            (list(a), list(b)) for a, b in data["required_merges"]
        ]

    def match(self, entity: Entity) -> tuple[str | None, list[str]]:
        """Entrée de référence d'une entité prédite (nom + alias).

        Retourne (entrée ou None, candidates si ambiguïté). En cas de matches
        multiples : première entrée dans l'ordre du fichier (spec)."""
        hits = set()
        for mention in [entity.name, *entity.aliases]:
            for key in _mention_keys(mention):
                if key in self.variant_index:
                    hits.add(self.variant_index[key])
        if not hits:
            return None, []
        ordered = sorted(hits, key=self.order.__getitem__)
        return ordered[0], ordered if len(ordered) > 1 else []


def load_reference(path: Path) -> Reference:
    return Reference(json.loads(path.read_text(encoding="utf-8")))


def _prf(n_correct: int, n_pred: int, n_covered: int, n_core: int) -> tuple:
    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_covered / n_core if n_core else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def _resolve_any(graph: KnowledgeGraph, mentions: list[str]) -> Entity | None:
    for m in mentions:
        e = graph.resolve(m)
        if e is not None:
            return e
    return None


def score_reference(graph: KnowledgeGraph, ref: Reference) -> dict:
    warnings = list(ref.variant_collisions)

    # --- entités : chaque prédite mappe vers au plus une entrée ---
    entry_hits: dict[str, int] = {}
    entity_entry: dict[str, str | None] = {}
    fp_entities: list[str] = []
    for e in graph.entities:
        entry, ambiguous = ref.match(e)
        if ambiguous:
            warnings.append(f"entité « {e.name} » matche plusieurs entrées : {ambiguous}")
        entity_entry[e.name] = entry
        if entry is None:
            fp_entities.append(e.name)
        else:
            entry_hits[entry] = entry_hits.get(entry, 0) + 1

    core_names = [e["name"] for e in ref.entries if e["level"] == "core"]
    covered = [n for n in core_names if n in entry_hits]
    n_pred = len(graph.entities)
    # doublons (entrée couverte 2+ fois) : nœuds de trop, exclus des correctes
    e_p, e_r, e_f1 = _prf(len(entry_hits), n_pred, len(covered), len(core_names))

    # --- relations : paires non orientées DISTINCTES du graphe prédit ---
    predicted_pairs: dict[frozenset, list[str]] = {}
    for r in graph.relations:
        ends = []
        for end in (r.source, r.target):
            resolved = graph.resolve(end)
            entry = entity_entry.get(resolved.name) if resolved else None
            # extrémité hors référence : marqueur « ? » (aucune entrée ne
            # commence par « ? »), la paire devient un faux positif
            ends.append(entry if entry is not None else "?" + normalize(end))
        predicted_pairs.setdefault(frozenset(ends), [r.source, r.target])

    tp_keys: set[frozenset] = set()
    n_neutral = 0
    fp_relations: list[list[str]] = []
    for pair, sample in predicted_pairs.items():
        if any(x.startswith("?") for x in pair):
            fp_relations.append(sample)
        elif pair in ref.core_pairs:
            tp_keys.add(pair)
        elif pair in ref.optional_pairs or any(ref.level[x] == "optional" for x in pair):
            n_neutral += 1
        else:
            fp_relations.append(sample)

    r_p, r_r, r_f1 = _prf(
        len(tp_keys) + n_neutral, len(predicted_pairs), len(tp_keys), len(ref.core_pairs)
    )
    missing_relations = sorted(
        sorted(p, key=ref.order.__getitem__) for p in ref.core_pairs - tp_keys
    )

    # --- fusions exigées : les deux groupes mènent à la même entité ---
    failed_merges = []
    for group_a, group_b in ref.required_merges:
        ea = _resolve_any(graph, group_a)
        eb = _resolve_any(graph, group_b)
        if ea is None or eb is None or ea is not eb:
            failed_merges.append([group_a, group_b])
    n_merges = len(ref.required_merges)

    return {
        "n_entities": n_pred,
        "n_relations": len(graph.relations),
        "n_predicted_pairs": len(predicted_pairs),
        "entity_precision": e_p,
        "entity_recall": e_r,
        "entity_f1": e_f1,
        "relation_precision": r_p,
        "relation_recall": r_r,
        "relation_f1": r_f1,
        "merges_ok": n_merges - len(failed_merges),
        "merges_total": n_merges,
        "merge_rate": f"{n_merges - len(failed_merges)}/{n_merges}",
        "failed_merges": failed_merges,
        "missing_entities": [n for n in core_names if n not in entry_hits],
        "false_positive_entities": sorted(fp_entities),
        "duplicate_entities": sorted(n for n, c in entry_hits.items() if c > 1),
        "missing_relations": missing_relations,
        "false_positive_relations": fp_relations,
        "warnings": warnings,
    }


def validate_reference(ref: Reference) -> list[str]:
    """Cohérence interne d'un fichier de référence (test de garde)."""
    errors = list(ref.variant_collisions)
    for entry in ref.entries:
        if entry["level"] not in VALID_LEVELS:
            errors.append(f"niveau invalide pour « {entry['name']} » : {entry['level']}")
        if not entry["variants"]:
            errors.append(f"aucun variant pour « {entry['name']} »")
    seen_pairs: set[frozenset] = set()
    for r in ref.data["relations"]:
        pair = r["pair"]
        if r["level"] not in VALID_LEVELS:
            errors.append(f"niveau invalide pour la paire {pair} : {r['level']}")
        if len(pair) != 2 or pair[0] == pair[1]:
            errors.append(f"paire invalide : {pair}")
        for name in pair:
            if name not in ref.order:
                errors.append(f"paire {pair} : entrée inconnue « {name} »")
        key = frozenset(pair)
        if key in seen_pairs:
            errors.append(f"paire dupliquée : {pair}")
        seen_pairs.add(key)
    for group_a, group_b in ref.required_merges:
        owners: set[str] = set()
        for mention in [*group_a, *group_b]:
            keys = [k for k in _mention_keys(mention) if k in ref.variant_index]
            if not keys:
                errors.append(f"fusion {[group_a, group_b]} : « {mention} » hors variants")
            else:
                owners.add(ref.variant_index[keys[0]])
        if len(owners) > 1:
            errors.append(
                f"fusion {[group_a, group_b]} : mentions dans des entrées "
                f"différentes {sorted(owners)}"
            )
    return errors
