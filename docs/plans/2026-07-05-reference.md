# Référence exhaustive P/R (axe 2) — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** mesurer précision ET rappel (entités, relations, fusions d'alias) contre une référence exhaustive, sur deux textes hors données d'entraînement.

**Architecture :** un module de scoring déterministe `benchmarks/reference_scoring.py` (chargé par importlib, motif `rescore.py`), deux fichiers de référence JSON déclaratifs, un harnais `bench_reference.py` (conventions `bench.py`), une extension de `rescore.py`. Spec : `docs/specs/2026-07-05-minerva-reference-design.md`.

**Tech stack :** Python 3.12, pydantic (modèle existant), pytest, Ollama local pour la campagne.

**Conventions repo :** tests dans `tests/`, code de bench dans `benchmarks/` (jamais dans `src/`), commits fréquents en français, `.venv/bin/python` / `.venv/bin/pytest`.

---

### Tâche 1 : scorer — appariement et métriques entités

**Files:**
- Create: `benchmarks/reference_scoring.py`
- Test: `tests/test_reference_scoring.py`

- [ ] **Step 1 : écrire les tests entités (échec attendu)**

Créer `tests/test_reference_scoring.py` :

```python
"""Tests du scoring exhaustif précision/rappel (benchmarks/reference_scoring.py)."""

import importlib.util
from pathlib import Path

from minerva.model import Entity, KnowledgeGraph, Relation

_BENCH = Path(__file__).parent.parent / "benchmarks"
_spec = importlib.util.spec_from_file_location(
    "reference_scoring", _BENCH / "reference_scoring.py"
)
scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring)

# Référence synthétique : 3 entités core, 1 optionnelle, 2 paires core,
# 1 paire optionnelle, 1 fusion exigée.
REF_DATA = {
    "text": "test.txt",
    "entities": [
        {"name": "Alice Vernet", "type": "personne",
         "variants": ["Alice Vernet", "Alice", "docteur Vernet"], "level": "core"},
        {"name": "Bruno Maillard", "type": "personne",
         "variants": ["Bruno Maillard", "Bruno"], "level": "core"},
        {"name": "Chaville", "type": "lieu", "variants": ["Chaville"], "level": "core"},
        {"name": "la Poste", "type": "organisation",
         "variants": ["la Poste", "Poste"], "level": "optional"},
    ],
    "required_merges": [[["Alice Vernet"], ["docteur Vernet"]]],
    "relations": [
        {"pair": ["Alice Vernet", "Bruno Maillard"], "level": "core"},
        {"pair": ["Alice Vernet", "Chaville"], "level": "core"},
        {"pair": ["Bruno Maillard", "Chaville"], "level": "optional"},
    ],
}


def make_ref():
    return scoring.Reference(REF_DATA)


def perfect_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne", aliases=["docteur Vernet"]))
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    g.add_entity(Entity(name="Chaville", type="lieu"))
    g.add_relation(Relation(name="connaît", source="Alice Vernet", target="Bruno Maillard"))
    g.add_relation(Relation(name="habite", source="Alice Vernet", target="Chaville"))
    return g


def test_graphe_parfait_entites():
    s = scoring.score_reference(perfect_graph(), make_ref())
    assert s["entity_precision"] == 1.0
    assert s["entity_recall"] == 1.0
    assert s["entity_f1"] == 1.0
    assert s["missing_entities"] == []
    assert s["false_positive_entities"] == []
    assert s["duplicate_entities"] == []


def test_entite_halucinee_compte_en_faux_positif():
    g = perfect_graph()
    g.add_entity(Entity(name="Zorglub", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["entity_precision"] == 0.75  # 3 matchées / 4 prédites
    assert s["entity_recall"] == 1.0
    assert s["false_positive_entities"] == ["Zorglub"]


def test_entite_optionnelle_neutre():
    # « la Poste » prédite : correcte en précision, absente du rappel.
    g = perfect_graph()
    g.add_entity(Entity(name="la Poste", type="organisation"))
    s = scoring.score_reference(g, make_ref())
    assert s["entity_precision"] == 1.0
    assert s["entity_recall"] == 1.0


def test_entite_manquante_baisse_le_rappel():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne"))
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["entity_recall"] == round(2 / 3, 3)
    assert s["missing_entities"] == ["Chaville"]
    assert s["entity_precision"] == 1.0


def test_doublon_non_fusionne_compte_en_faux_positif():
    # « docteur Vernet » créé comme entité séparée d'« Alice Vernet » (sans
    # alias posé, la résolution du graphe ne les relie pas) : même entrée de
    # référence couverte deux fois -> 1 nœud de trop.
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne"))
    g.add_entity(Entity(name="docteur Vernet", type="personne"))
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    g.add_entity(Entity(name="Chaville", type="lieu"))
    assert len(g.entities) == 4  # pas de fusion : « vernet » nu n'est indexé nulle part
    s = scoring.score_reference(g, make_ref())
    assert s["entity_precision"] == 0.75  # 3 entrées distinctes / 4 prédites
    assert s["entity_recall"] == 1.0
    assert s["duplicate_entities"] == ["Alice Vernet"]


def test_match_via_forme_sans_titre():
    # « commissaire Bruno Maillard » prédite : strip_title -> « Bruno Maillard ».
    g = KnowledgeGraph()
    g.add_entity(Entity(name="commissaire Bruno Maillard", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["false_positive_entities"] == []
    assert "Bruno Maillard" not in s["missing_entities"]


def test_graphe_vide():
    s = scoring.score_reference(KnowledgeGraph(), make_ref())
    assert s["entity_precision"] == 0.0
    assert s["entity_recall"] == 0.0
    assert s["entity_f1"] == 0.0
```

Note sur `test_doublon…` : la construction évite les replis de fusion du
graphe — « Alice Vernet » ne strippe aucun titre (donc « vernet » nu n'entre
pas dans l'index des formes dépouillées) et « docteur Vernet » strippe vers
« vernet », absent des entités : les deux restent séparées. Côté scorer,
les deux matchent l'entrée « Alice Vernet » (variants « Alice Vernet » et
« docteur Vernet ») → doublon.

- [ ] **Step 2 : vérifier l'échec**

Run : `.venv/bin/pytest tests/test_reference_scoring.py -x -q`
Attendu : erreur d'import (`reference_scoring.py` inexistant).

- [ ] **Step 3 : implémenter le module (entités seulement)**

Créer `benchmarks/reference_scoring.py` :

```python
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

    return {
        "n_entities": n_pred,
        "n_relations": len(graph.relations),
        "entity_precision": e_p,
        "entity_recall": e_r,
        "entity_f1": e_f1,
        "missing_entities": [n for n in core_names if n not in entry_hits],
        "false_positive_entities": sorted(fp_entities),
        "duplicate_entities": sorted(n for n, c in entry_hits.items() if c > 1),
        "warnings": warnings,
    }
```

- [ ] **Step 4 : vérifier le vert**

Run : `.venv/bin/pytest tests/test_reference_scoring.py -x -q`
Attendu : tous les tests entités PASS. Si `test_doublon…` échoue sur
`len(g.entities) == 4`, appliquer la construction de repli documentée dans
le test (deux entités distinctes mappant la même entrée), relancer.

- [ ] **Step 5 : commit**

```bash
git add benchmarks/reference_scoring.py tests/test_reference_scoring.py
git commit -m "bench: scoring exhaustif — appariement et P/R entités (axe 2)"
```

---

### Tâche 2 : scorer — relations (paires non orientées)

**Files:**
- Modify: `benchmarks/reference_scoring.py`
- Test: `tests/test_reference_scoring.py`

- [ ] **Step 1 : ajouter les tests relations (échec attendu)**

Ajouter à `tests/test_reference_scoring.py` :

```python
def test_graphe_parfait_relations():
    s = scoring.score_reference(perfect_graph(), make_ref())
    assert s["relation_precision"] == 1.0
    assert s["relation_recall"] == 1.0
    assert s["missing_relations"] == []
    assert s["false_positive_relations"] == []


def test_relation_manquante_baisse_le_rappel():
    g = perfect_graph()
    # ne garder que Alice-Bruno : reconstruire sans Alice-Chaville
    g2 = KnowledgeGraph()
    g2.add_entity(Entity(name="Alice Vernet", type="personne", aliases=["docteur Vernet"]))
    g2.add_entity(Entity(name="Bruno Maillard", type="personne"))
    g2.add_entity(Entity(name="Chaville", type="lieu"))
    g2.add_relation(Relation(name="connaît", source="Alice Vernet", target="Bruno Maillard"))
    s = scoring.score_reference(g2, make_ref())
    assert s["relation_recall"] == 0.5
    assert s["missing_relations"] == [["Alice Vernet", "Chaville"]]
    assert s["relation_precision"] == 1.0


def test_paire_optionnelle_neutre():
    g = perfect_graph()
    g.add_relation(Relation(name="voisin", source="Bruno Maillard", target="Chaville"))
    s = scoring.score_reference(g, make_ref())
    assert s["relation_precision"] == 1.0  # (2 TP + 1 neutre) / 3
    assert s["relation_recall"] == 1.0


def test_relation_vers_entite_optionnelle_neutre():
    g = perfect_graph()
    g.add_entity(Entity(name="la Poste", type="organisation"))
    g.add_relation(Relation(name="travaille à", source="Bruno Maillard", target="la Poste"))
    s = scoring.score_reference(g, make_ref())
    assert s["relation_precision"] == 1.0
    assert s["relation_recall"] == 1.0


def test_paire_core_core_hors_reference_faux_positif():
    g = perfect_graph()
    g.add_relation(Relation(name="hante", source="Chaville", target="Bruno Maillard"))
    # {Bruno, Chaville} est optionnelle -> neutre ; il faut une paire absente :
    # Chaville-Chaville n'existe pas ; on passe par une entité hallucinée.
    g.add_entity(Entity(name="Zorglub", type="personne"))
    g.add_relation(Relation(name="menace", source="Zorglub", target="Alice Vernet"))
    s = scoring.score_reference(g, make_ref())
    # paires : {A,B} TP, {A,C} TP, {B,C} neutre, {Zorglub,A} FP -> 3/4
    assert s["relation_precision"] == 0.75
    assert s["false_positive_relations"] == [["Zorglub", "Alice Vernet"]]


def test_sens_et_predicat_ignores_et_multipredicats_dedupliques():
    g = perfect_graph()
    # sens inverse + deuxième prédicat sur la même paire : toujours 1 paire
    g.add_relation(Relation(name="emploie", source="Bruno Maillard", target="Alice Vernet"))
    s = scoring.score_reference(g, make_ref())
    assert s["relation_precision"] == 1.0
    assert s["relation_recall"] == 1.0
```

- [ ] **Step 2 : vérifier l'échec**

Run : `.venv/bin/pytest tests/test_reference_scoring.py -x -q`
Attendu : `KeyError: 'relation_precision'` (ou assertion) sur les nouveaux tests.

- [ ] **Step 3 : implémenter le volet relations**

Dans `score_reference`, avant le `return`, ajouter :

```python
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
```

et enrichir le dict retourné :

```python
        "n_predicted_pairs": len(predicted_pairs),
        "relation_precision": r_p,
        "relation_recall": r_r,
        "relation_f1": r_f1,
        "missing_relations": missing_relations,
        "false_positive_relations": fp_relations,
```

Note : une paire réflexive (deux extrémités mappées sur la même entrée,
`frozenset` de taille 1) n'est jamais dans la référence (paires validées de
taille 2) → faux positif, comportement voulu (relation entre doublons).

- [ ] **Step 4 : vérifier le vert**

Run : `.venv/bin/pytest tests/test_reference_scoring.py -x -q`
Attendu : PASS (13 tests).

- [ ] **Step 5 : commit**

```bash
git add benchmarks/reference_scoring.py tests/test_reference_scoring.py
git commit -m "bench: scoring exhaustif — P/R relations en paires non orientées"
```

---

### Tâche 3 : scorer — fusions exigées et validation de référence

**Files:**
- Modify: `benchmarks/reference_scoring.py`
- Test: `tests/test_reference_scoring.py`

- [ ] **Step 1 : tests fusions + validation (échec attendu)**

Ajouter à `tests/test_reference_scoring.py` :

```python
def test_fusion_reussie():
    s = scoring.score_reference(perfect_graph(), make_ref())
    assert s["merge_rate"] == "1/1"
    assert s["failed_merges"] == []


def test_fusion_ratee():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne"))  # sans alias
    g.add_entity(Entity(name="Vernet", type="personne", aliases=["docteur Vernet"]))
    s = scoring.score_reference(g, make_ref())
    assert s["merge_rate"] == "0/1"
    assert s["failed_merges"] == [[["Alice Vernet"], ["docteur Vernet"]]]


def test_fusion_mention_absente_du_graphe():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["merge_rate"] == "0/1"


def test_validation_reference_saine():
    assert scoring.validate_reference(make_ref()) == []


def test_validation_detecte_les_incoherences():
    bad = {
        "text": "bad.txt",
        "entities": [
            {"name": "A", "type": "personne", "variants": ["A", "Double"], "level": "core"},
            {"name": "B", "type": "personne", "variants": ["B", "Double"], "level": "core"},
            {"name": "C", "type": "personne", "variants": [], "level": "exotique"},
        ],
        "required_merges": [[["Fantôme"], ["A"]], [["A"], ["B"]]],
        "relations": [
            {"pair": ["A", "Inconnu"], "level": "core"},
            {"pair": ["A", "A"], "level": "core"},
            {"pair": ["A", "B"], "level": "core"},
            {"pair": ["B", "A"], "level": "optional"},
        ],
    }
    errors = scoring.validate_reference(scoring.Reference(bad))
    text = "\n".join(errors)
    assert "Double" in text          # variant en collision
    assert "exotique" in text        # niveau invalide
    assert "variants" in text or "aucun variant" in text  # C sans variant
    assert "Inconnu" in text         # paire vers entrée inexistante
    assert "['A', 'A']" in text      # paire réflexive
    assert "dupliquée" in text       # {A,B} deux fois
    assert "Fantôme" in text         # fusion hors variants
    assert "entrées différentes" in text  # fusion A/B inter-entrées
```

- [ ] **Step 2 : vérifier l'échec**

Run : `.venv/bin/pytest tests/test_reference_scoring.py -x -q`
Attendu : `KeyError: 'merge_rate'` puis `AttributeError: validate_reference`.

- [ ] **Step 3 : implémenter fusions + validation**

Dans `reference_scoring.py`, ajouter la fonction module :

```python
def _resolve_any(graph: KnowledgeGraph, mentions: list[str]) -> Entity | None:
    for m in mentions:
        e = graph.resolve(m)
        if e is not None:
            return e
    return None
```

Dans `score_reference`, avant le `return` :

```python
    # --- fusions exigées : les deux groupes mènent à la même entité ---
    failed_merges = []
    for group_a, group_b in ref.required_merges:
        ea = _resolve_any(graph, group_a)
        eb = _resolve_any(graph, group_b)
        if ea is None or eb is None or ea is not eb:
            failed_merges.append([group_a, group_b])
    n_merges = len(ref.required_merges)
```

et dans le dict retourné :

```python
        "merges_ok": n_merges - len(failed_merges),
        "merges_total": n_merges,
        "merge_rate": f"{n_merges - len(failed_merges)}/{n_merges}",
        "failed_merges": failed_merges,
```

Ajouter la validation (fin de module) :

```python
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
```

- [ ] **Step 4 : vérifier le vert, puis la suite complète**

Run : `.venv/bin/pytest tests/test_reference_scoring.py -q` puis `.venv/bin/pytest -q`
Attendu : tout PASS (aucune régression ailleurs).

- [ ] **Step 5 : commit**

```bash
git add benchmarks/reference_scoring.py tests/test_reference_scoring.py
git commit -m "bench: scoring exhaustif — fusions exigées et validation de référence"
```

---

### Tâche 4 : texte inédit + les deux références exhaustives + test de garde

**Files:**
- Create: `benchmarks/reference_texte.txt`
- Create: `benchmarks/reference_reference_texte.json`
- Create: `benchmarks/reference_timeline_texte.json`
- Test: `tests/test_references.py`

- [ ] **Step 1 : écrire le texte inédit**

Créer `benchmarks/reference_texte.txt` avec EXACTEMENT ce contenu (écrit pour
ce bench, hors données d'entraînement ; identité d'emprunt Sérac=Rivière,
titres « commissaire »/« docteur », diminutif « Margot », deux fils
narratifs — Valsonne et Besançon — croisés) :

```
Le cartographe de Valsonne

Antoine Sérac était arrivé à Valsonne par un matin de brume, trois ans
plus tôt, avec pour tout bagage une malle d'instruments de cuivre. Il
avait loué l'ancienne remise de la rue des Tanneurs et y avait ouvert un
atelier de cartographie. La Société de géographie de Valsonne, séduite
par la précision de ses relevés, venait de lui commander une carte des
hauteurs du mont Chauvel, dont personne n'avait encore levé les pentes
nord.

À deux rues de là, place aux Herbes, Marguerite Aubanel tenait la
librairie du Méridien. Tout le monde l'appelait Margot. Elle mettait de
côté pour Sérac les atlas anciens et les récits de voyage ; il venait
les chercher le samedi, restait une heure, parfois deux. Son frère
cadet, Lucien Aubanel, qui s'ennuyait derrière le comptoir de la
librairie, obtint au printemps d'entrer à l'atelier comme apprenti :
Sérac lui apprit à tendre le papier, puis à tenir la plume.

Le commissaire Félix Brossard, lui, n'aimait pas les hommes sans passé.
Il avait remarqué que le cartographe ne parlait jamais de ses années de
formation, et qu'aucun confrère ne le connaissait. Or Brossard avait de
la mémoire : dix ans plus tôt, toute la province avait parlé de
l'effondrement du pont de la Frène, à Saint-Elme, et du géomètre dont
les relevés truqués avaient été jugés responsables. L'homme, condamné,
avait purgé sa peine, puis s'était volatilisé.

À Besançon, où les toits d'ocre descendent vers les méandres de la
Loue, Irène Vaneau classait les minutes anciennes du tribunal. En
reliant un carton d'archives, elle rouvrit par hasard le dossier de
Théo Rivière, ce géomètre de Saint-Elme condamné pour l'affaire du pont
de la Frène. Une note du greffe signalait que Rivière, à sa libération,
avait retiré des dépôts une malle d'instruments de cuivre. Troublée,
Irène en parla dans une lettre à son frère, le docteur Paul Vaneau, qui
exerçait la médecine à Valsonne ; puis, sur son conseil, elle écrivit
au commissaire Brossard.

Cet hiver-là, Margot prit froid à l'inventaire et garda le lit une
semaine. Le docteur Vaneau, qui la soignait, trouva un soir Sérac à son
chevet, en train de lui lire un portulan comme on lit un roman. C'est à
elle que Sérac dit la vérité, un soir de décembre : il s'appelait Théo
Rivière ; c'étaient ses calques qu'un entrepreneur pressé avait
falsifiés autrefois, mais la faute était retombée sur lui seul ; il
avait pris à sa sortie le nom d'Antoine Sérac pour pouvoir tenir encore
un compas. Margot garda le secret et ne changea rien à ses samedis.

La lettre de Besançon arriva en janvier. Brossard convoqua le
cartographe et posa devant lui le dossier Rivière. Sérac ne nia rien ;
il déplia seulement, à côté du dossier, la carte achevée du mont
Chauvel, cotée pente par pente, et proposa au commissaire d'en faire
vérifier chaque point par la Société de géographie. Brossard fit
vérifier. Au printemps, quand la carte revint sans une seule
correction, il classa le dossier de sa propre main. La carte fut
gravée, et l'on peut y lire, sous le cartouche, une double signature :
« A. Sérac — Th. Rivière ».
```

- [ ] **Step 2 : écrire la référence exhaustive du nouveau texte**

Créer `benchmarks/reference_reference_texte.json` (13 entités core,
4 optionnelles, 13 paires core, fusions : identité d'emprunt, diminutif,
deux titres) :

```json
{
  "text": "reference_texte.txt",
  "entities": [
    {"name": "Antoine Sérac", "type": "personne",
     "variants": ["Antoine Sérac", "Sérac", "A. Sérac", "Théo Rivière", "Rivière", "Th. Rivière", "le cartographe"],
     "level": "core"},
    {"name": "Marguerite Aubanel", "type": "personne",
     "variants": ["Marguerite Aubanel", "Marguerite", "Margot"],
     "level": "core"},
    {"name": "Lucien Aubanel", "type": "personne",
     "variants": ["Lucien Aubanel", "Lucien"],
     "level": "core"},
    {"name": "Félix Brossard", "type": "personne",
     "variants": ["Félix Brossard", "Brossard", "commissaire Brossard", "le commissaire Félix Brossard"],
     "level": "core"},
    {"name": "Paul Vaneau", "type": "personne",
     "variants": ["Paul Vaneau", "docteur Paul Vaneau", "docteur Vaneau", "Paul"],
     "level": "core"},
    {"name": "Irène Vaneau", "type": "personne",
     "variants": ["Irène Vaneau", "Irène"],
     "level": "core"},
    {"name": "Valsonne", "type": "lieu", "variants": ["Valsonne"], "level": "core"},
    {"name": "Besançon", "type": "lieu", "variants": ["Besançon"], "level": "core"},
    {"name": "Saint-Elme", "type": "lieu", "variants": ["Saint-Elme"], "level": "core"},
    {"name": "Société de géographie de Valsonne", "type": "organisation",
     "variants": ["Société de géographie de Valsonne", "Société de géographie", "la Société de géographie"],
     "level": "core"},
    {"name": "librairie du Méridien", "type": "organisation",
     "variants": ["librairie du Méridien", "la librairie du Méridien", "le Méridien"],
     "level": "core"},
    {"name": "pont de la Frène", "type": "lieu",
     "variants": ["pont de la Frène", "le pont de la Frène", "la Frène"],
     "level": "core"},
    {"name": "mont Chauvel", "type": "lieu",
     "variants": ["mont Chauvel", "le mont Chauvel", "Chauvel"],
     "level": "core"},
    {"name": "rue des Tanneurs", "type": "lieu",
     "variants": ["rue des Tanneurs", "la rue des Tanneurs"],
     "level": "optional"},
    {"name": "place aux Herbes", "type": "lieu",
     "variants": ["place aux Herbes", "la place aux Herbes"],
     "level": "optional"},
    {"name": "la Loue", "type": "lieu", "variants": ["la Loue", "Loue"], "level": "optional"},
    {"name": "tribunal de Besançon", "type": "organisation",
     "variants": ["tribunal de Besançon", "le tribunal de Besançon", "le tribunal", "tribunal"],
     "level": "optional"}
  ],
  "required_merges": [
    [["Antoine Sérac", "Sérac"], ["Théo Rivière", "Rivière"]],
    [["Marguerite Aubanel"], ["Margot"]],
    [["Paul Vaneau", "le docteur Paul Vaneau"], ["le docteur Vaneau", "docteur Vaneau"]],
    [["Félix Brossard"], ["le commissaire Brossard", "commissaire Brossard"]]
  ],
  "relations": [
    {"pair": ["Antoine Sérac", "Valsonne"], "level": "core"},
    {"pair": ["Antoine Sérac", "Société de géographie de Valsonne"], "level": "core"},
    {"pair": ["Antoine Sérac", "mont Chauvel"], "level": "core"},
    {"pair": ["Marguerite Aubanel", "librairie du Méridien"], "level": "core"},
    {"pair": ["Antoine Sérac", "Marguerite Aubanel"], "level": "core"},
    {"pair": ["Marguerite Aubanel", "Lucien Aubanel"], "level": "core"},
    {"pair": ["Antoine Sérac", "Lucien Aubanel"], "level": "core"},
    {"pair": ["Félix Brossard", "Antoine Sérac"], "level": "core"},
    {"pair": ["Antoine Sérac", "pont de la Frène"], "level": "core"},
    {"pair": ["Antoine Sérac", "Saint-Elme"], "level": "core"},
    {"pair": ["Irène Vaneau", "Paul Vaneau"], "level": "core"},
    {"pair": ["Irène Vaneau", "Félix Brossard"], "level": "core"},
    {"pair": ["Paul Vaneau", "Marguerite Aubanel"], "level": "core"},
    {"pair": ["Paul Vaneau", "Valsonne"], "level": "optional"},
    {"pair": ["Paul Vaneau", "Besançon"], "level": "optional"},
    {"pair": ["Paul Vaneau", "Antoine Sérac"], "level": "optional"},
    {"pair": ["Irène Vaneau", "Besançon"], "level": "optional"},
    {"pair": ["Irène Vaneau", "Saint-Elme"], "level": "optional"},
    {"pair": ["Irène Vaneau", "pont de la Frène"], "level": "optional"},
    {"pair": ["Irène Vaneau", "Antoine Sérac"], "level": "optional"},
    {"pair": ["Félix Brossard", "Valsonne"], "level": "optional"},
    {"pair": ["Félix Brossard", "pont de la Frène"], "level": "optional"},
    {"pair": ["Félix Brossard", "Société de géographie de Valsonne"], "level": "optional"},
    {"pair": ["Lucien Aubanel", "librairie du Méridien"], "level": "optional"},
    {"pair": ["Lucien Aubanel", "Valsonne"], "level": "optional"},
    {"pair": ["Marguerite Aubanel", "Valsonne"], "level": "optional"},
    {"pair": ["pont de la Frène", "Saint-Elme"], "level": "optional"},
    {"pair": ["Société de géographie de Valsonne", "Valsonne"], "level": "optional"},
    {"pair": ["mont Chauvel", "Valsonne"], "level": "optional"}
  ]
}
```

- [ ] **Step 3 : écrire la référence exhaustive de timeline_texte.txt**

Créer `benchmarks/reference_timeline_texte.json` (relire
`benchmarks/timeline_texte.txt` pour contrôler l'inventaire — chaque nom
propre du texte doit figurer ci-dessous) :

```json
{
  "text": "timeline_texte.txt",
  "entities": [
    {"name": "Élise Chardon", "type": "personne",
     "variants": ["Élise Chardon", "Élise"], "level": "core"},
    {"name": "Bastien Malot", "type": "personne",
     "variants": ["Bastien Malot", "Bastien"], "level": "core"},
    {"name": "Aurélien Chardon", "type": "personne",
     "variants": ["Aurélien Chardon", "Aurélien", "grand-père Aurélien"], "level": "core"},
    {"name": "Camille Roche", "type": "personne",
     "variants": ["Camille Roche", "Camille"], "level": "core"},
    {"name": "rue des Grilles", "type": "lieu",
     "variants": ["rue des Grilles", "la rue des Grilles", "atelier de la rue des Grilles", "l'atelier de la rue des Grilles"],
     "level": "core"},
    {"name": "Aubervilliers", "type": "lieu", "variants": ["Aubervilliers"], "level": "core"},
    {"name": "Mirecourt", "type": "lieu", "variants": ["Mirecourt"], "level": "core"},
    {"name": "Lyon", "type": "lieu", "variants": ["Lyon"], "level": "core"},
    {"name": "école Boulle", "type": "organisation",
     "variants": ["école Boulle", "l'école Boulle"], "level": "optional"},
    {"name": "la Croix-Rousse", "type": "lieu",
     "variants": ["la Croix-Rousse", "Croix-Rousse"], "level": "optional"},
    {"name": "la Saône", "type": "lieu", "variants": ["la Saône", "Saône"], "level": "optional"},
    {"name": "canif de buis", "type": "objet",
     "variants": ["canif de buis", "le canif de buis", "canif", "le canif", "couteau à manche de buis", "canif d'Aurélien"],
     "level": "optional"}
  ],
  "required_merges": [
    [["Élise Chardon"], ["Élise"]],
    [["Bastien Malot"], ["Bastien"]],
    [["Aurélien Chardon"], ["Aurélien"]]
  ],
  "relations": [
    {"pair": ["Élise Chardon", "Bastien Malot"], "level": "core"},
    {"pair": ["Élise Chardon", "Aurélien Chardon"], "level": "core"},
    {"pair": ["Élise Chardon", "rue des Grilles"], "level": "core"},
    {"pair": ["Bastien Malot", "rue des Grilles"], "level": "core"},
    {"pair": ["Aurélien Chardon", "Mirecourt"], "level": "core"},
    {"pair": ["Élise Chardon", "Mirecourt"], "level": "core"},
    {"pair": ["Camille Roche", "Lyon"], "level": "core"},
    {"pair": ["Bastien Malot", "Camille Roche"], "level": "core"},
    {"pair": ["rue des Grilles", "Aubervilliers"], "level": "optional"},
    {"pair": ["Élise Chardon", "Aubervilliers"], "level": "optional"},
    {"pair": ["Bastien Malot", "Aubervilliers"], "level": "optional"},
    {"pair": ["Camille Roche", "rue des Grilles"], "level": "optional"}
  ]
}
```

Note : les fusions timeline sont volontairement des paires prénom/nom
complet — moins dures que l'identité d'emprunt, elles mesurent la pose
d'alias de base. Les mentions des `required_merges` doivent rester des
sous-ensembles des `variants` (la validation l'exige).

- [ ] **Step 4 : test de garde des vrais fichiers**

Créer `tests/test_references.py` :

```python
"""Garde de cohérence des fichiers de référence exhaustive (axe 2)."""

import importlib.util
from pathlib import Path

import pytest

_BENCH = Path(__file__).parent.parent / "benchmarks"
_spec = importlib.util.spec_from_file_location(
    "reference_scoring", _BENCH / "reference_scoring.py"
)
scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring)

REFERENCES = {
    "reference_reference_texte.json": "reference_texte.txt",
    "reference_timeline_texte.json": "timeline_texte.txt",
}


@pytest.mark.parametrize("ref_name", sorted(REFERENCES))
def test_reference_coherente(ref_name):
    ref = scoring.load_reference(_BENCH / ref_name)
    assert scoring.validate_reference(ref) == []


@pytest.mark.parametrize("ref_name", sorted(REFERENCES))
def test_reference_pointe_vers_son_texte(ref_name):
    ref = scoring.load_reference(_BENCH / ref_name)
    assert ref.data["text"] == REFERENCES[ref_name]
    assert (_BENCH / ref.data["text"]).exists()


@pytest.mark.parametrize("ref_name", sorted(REFERENCES))
def test_variants_presents_dans_le_texte(ref_name):
    """Chaque entrée a au moins un variant littéralement présent dans le
    texte (une référence exhaustive décrit le texte, pas un canon externe)."""
    ref = scoring.load_reference(_BENCH / ref_name)
    text = (_BENCH / ref.data["text"]).read_text(encoding="utf-8")
    flat = " ".join(text.split()).casefold()
    for entry in ref.entries:
        found = any(" ".join(v.split()).casefold() in flat for v in entry["variants"])
        assert found, f"{ref_name} : aucun variant de « {entry['name']} » dans le texte"
```

- [ ] **Step 5 : lancer et corriger jusqu'au vert**

Run : `.venv/bin/pytest tests/test_references.py -q`
Attendu : PASS. En cas d'échec de `test_variants_presents_dans_le_texte`
(variant reformulé), corriger le JSON — jamais le test.

- [ ] **Step 6 : relecture d'exhaustivité (manuelle, obligatoire)**

Relire les deux textes phrase par phrase et vérifier que CHAQUE nom propre
apparaît dans le JSON correspondant (core ou optional), et que chaque
relation explicite entre deux entités core est soit une paire core, soit
une paire optional. Critère de succès n° 2 de la spec.

- [ ] **Step 7 : commit**

```bash
git add benchmarks/reference_texte.txt benchmarks/reference_reference_texte.json \
        benchmarks/reference_timeline_texte.json tests/test_references.py
git commit -m "bench: texte inédit « Le cartographe de Valsonne » + références exhaustives des deux textes"
```

---

### Tâche 5 : harnais bench_reference.py

**Files:**
- Create: `benchmarks/bench_reference.py`

Pas de test automatisé (bench manuel appelant Ollama), mais un dry-run de
scoring sur graphe sauvegardé existant valide le câblage avant la campagne.

- [ ] **Step 1 : écrire le harnais**

Créer `benchmarks/bench_reference.py` :

```python
"""Bench précision/rappel contre les références exhaustives (axe 2).

Pour chaque (texte, pipeline, modèle) : extraction (temp. par défaut), puis
si pipeline « actee » : complétude + canonicalisation (temp 0, reco du
rapport). Scores P/R/F1 entités, relations, fusions — reference_scoring.py.

Usage :
    .venv/bin/python benchmarks/bench_reference.py [modèle ...] \
        [--text reference|timeline|all] [--pipeline nue|actee|all] [--runs N]
Prérequis : serveur Ollama local sur http://localhost:11434.
"""

import argparse
import datetime
import importlib.util
import json
import statistics
import time
import traceback
from pathlib import Path

from minerva.extraction import extract_graph
from minerva.llm.openai_backend import OpenAIBackend
from minerva.refine import canonicalize_graph, complete_graph

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location(
    "reference_scoring", HERE / "reference_scoring.py"
)
_scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring)

OLLAMA = "http://localhost:11434/v1"
CHUNK_SIZE = 700  # plusieurs chunks : cohérence inter-chunks testée
DEFAULT_MODELS = ["gpt-oss:120b", "qwen3-coder-next:latest"]
TEXTS = {"reference": "reference_texte.txt", "timeline": "timeline_texte.txt"}
PIPELINES = ("nue", "actee")


def backend(model: str, temperature: float | None = None) -> OpenAIBackend:
    return OpenAIBackend(model=model, base_url=OLLAMA, temperature=temperature)


def run_one(model: str, text: str, pipeline: str, temperature: float | None,
            label: str) -> tuple[dict, "KnowledgeGraph"]:
    t0 = time.monotonic()
    graph = extract_graph(
        text, backend(model, temperature), chunk_size=CHUNK_SIZE,
        on_progress=lambda d, t: print(f"  {label} chunk {d}/{t}", flush=True),
    )
    detail = ""
    if pipeline == "actee":
        n = complete_graph(graph, text, backend(model, temperature=0))
        before = len(graph.entities)
        graph = canonicalize_graph(graph, backend(model, temperature=0))
        detail = f"complétude +{n}, canon {before}->{len(graph.entities)}"
    entry = {"time_s": round(time.monotonic() - t0, 1), "detail": detail}
    return entry, graph


AGG_KEYS = (
    "time_s", "entity_precision", "entity_recall", "entity_f1",
    "relation_precision", "relation_recall", "relation_f1",
)


def aggregate(per_run: list[dict]) -> dict:
    agg: dict = {"per_run": per_run}
    for key in AGG_KEYS:
        values = [r[key] for r in per_run]
        agg[f"{key}_mean"] = round(statistics.mean(values), 3)
        agg[f"{key}_std"] = round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
    ok = sum(r["merges_ok"] for r in per_run)
    total = sum(r["merges_total"] for r in per_run)
    agg["merge_rate"] = f"{ok}/{total}"
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench P/R contre référence exhaustive")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--text", choices=[*TEXTS, "all"], default="reference")
    parser.add_argument("--pipeline", choices=[*PIPELINES, "all"], default="nue")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    models = args.models or DEFAULT_MODELS
    texts = list(TEXTS) if args.text == "all" else [args.text]
    pipelines = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]

    out_dir = HERE / "results" / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "reference_results.json"
    previous = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists() else []
    )

    def key(e: dict) -> tuple:
        return (e.get("model"), e.get("temperature"), e.get("runs"),
                e.get("text"), e.get("pipeline"))

    new_keys = {(m, args.temperature, args.runs, t, p)
                for m in models for t in texts for p in pipelines}
    results = [e for e in previous if key(e) not in new_keys]

    for model in models:
        # échauffement : charge le modèle hors chrono
        backend(model)._client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Réponds : ok"}]
        )
        safe = model.replace(":", "_").replace("/", "_")
        for text_key in texts:
            text = (HERE / TEXTS[text_key]).read_text(encoding="utf-8")
            ref = _scoring.load_reference(
                HERE / f"reference_{TEXTS[text_key].removesuffix('.txt')}.json"
            )
            for pipeline in pipelines:
                entry = {"model": model, "temperature": args.temperature,
                         "runs": args.runs, "text": text_key, "pipeline": pipeline}
                print(f"=== {model} / {text_key} / {pipeline} "
                      f"(runs={args.runs}) ===", flush=True)
                try:
                    per_run = []
                    for i in range(1, args.runs + 1):
                        label = f"{model} {text_key}/{pipeline} run {i}"
                        run_entry, graph = run_one(
                            model, text, pipeline, args.temperature, label
                        )
                        run_entry.update(_scoring.score_reference(graph, ref))
                        per_run.append(run_entry)
                        suffix = f"_run{i}" if args.runs > 1 else ""
                        (out_dir / f"ref_{text_key}_{pipeline}_{safe}{suffix}.json").write_text(
                            json.dumps(graph.to_dict(), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        print(json.dumps(
                            {k: run_entry[k] for k in (
                                "time_s", "entity_precision", "entity_recall",
                                "relation_precision", "relation_recall", "merge_rate")},
                            ensure_ascii=False), flush=True)
                    entry.update(per_run[0] if args.runs == 1 else aggregate(per_run))
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                results.append(entry)

    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TERMINÉ -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : dry-run du câblage scoring sans LLM**

Valider le chemin référence + scoring sur un graphe déjà sauvegardé :

```bash
.venv/bin/python - <<'EOF'
import importlib.util, json
from pathlib import Path
HERE = Path("benchmarks")
spec = importlib.util.spec_from_file_location("rs", HERE / "reference_scoring.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
spec2 = importlib.util.spec_from_file_location("rescore", HERE / "rescore.py")
rescore = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rescore)
ref = rs.load_reference(HERE / "reference_timeline_texte.json")
graph_path = sorted((HERE / "results/2026-07-04").glob("*.json"))
# prendre un graphe timeline sauvegardé de l'addendum 6
candidates = [p for p in graph_path if "timeline" in p.name or "pipeline_" in p.name]
g, skipped = rescore.load_graph(candidates[0])
print(candidates[0].name, json.dumps(rs.score_reference(g, ref), ensure_ascii=False, indent=1))
EOF
```

Attendu : un dict complet de métriques plausibles (P/R entre 0 et 1, listes
missing/faux positifs cohérentes avec le graphe inspecté), aucune exception.
Si aucun graphe timeline n'existe dans `results/2026-07-04`, adapter le
filtre `candidates` au listing réel du dossier.

- [ ] **Step 3 : vérifier la CLI sans exécuter de LLM**

Run : `.venv/bin/python benchmarks/bench_reference.py --help`
Attendu : l'aide s'affiche (imports et syntaxe valides).

- [ ] **Step 4 : commit**

```bash
git add benchmarks/bench_reference.py
git commit -m "bench: harnais bench_reference — P/R exhaustifs par texte × pipeline"
```

---

### Tâche 6 : extension de rescore.py

**Files:**
- Modify: `benchmarks/rescore.py`

- [ ] **Step 1 : ajouter le rescoring des graphes de référence**

Dans `benchmarks/rescore.py`, après le chargement de `bench.py` (ligne ~26),
charger aussi le scoring exhaustif :

```python
_spec_ref = importlib.util.spec_from_file_location(
    "reference_scoring", HERE / "reference_scoring.py"
)
_ref_scoring = importlib.util.module_from_spec(_spec_ref)
_spec_ref.loader.exec_module(_ref_scoring)
```

Ajouter avant `main` :

```python
def rescore_reference(results_dir: Path) -> None:
    """Re-score les graphes ref_*.json contre les références exhaustives
    courantes. Une entrée par graphe (modèle × texte × pipeline × run) —
    pas de ré-agrégation : sortie d'inspection."""
    refs = {
        key: _ref_scoring.load_reference(
            HERE / f"reference_{name.removesuffix('.txt')}.json"
        )
        for key, name in (("reference", "reference_texte.txt"),
                          ("timeline", "timeline_texte.txt"))
    }
    rescored = []
    for graph_path in sorted(results_dir.glob("ref_*.json")):
        text_key = graph_path.name.split("_")[1]
        if text_key not in refs:
            print(f"!! texte inconnu pour {graph_path.name}")
            continue
        graph, skipped = load_graph(graph_path)
        entry = {"graph": graph_path.name, "skipped_records": skipped}
        entry.update(_ref_scoring.score_reference(graph, refs[text_key]))
        rescored.append(entry)
        print(f"== {graph_path.name}: eP={entry['entity_precision']} "
              f"eR={entry['entity_recall']} rP={entry['relation_precision']} "
              f"rR={entry['relation_recall']} fusions={entry['merge_rate']}")
    out = results_dir / "reference_results_rescored.json"
    out.write_text(json.dumps(rescored, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
```

Et dans `main`, tout au début :

```python
    if any(results_dir.glob("ref_*.json")):
        rescore_reference(results_dir)
    results_path = results_dir / "bench_results.json"
    if not results_path.exists():
        return
```

(le reste de `main` inchangé — un dossier peut contenir les deux familles).

- [ ] **Step 2 : vérifier sur un dossier existant (non-régression)**

Run : `.venv/bin/python benchmarks/rescore.py results/2026-07-04`
Attendu : comportement identique à avant (pas de `ref_*.json` dans ce
dossier → seule la branche historique tourne, mêmes sorties qu'avant).

- [ ] **Step 3 : commit**

```bash
git add benchmarks/rescore.py
git commit -m "bench: rescore des graphes de référence exhaustive (ref_*.json)"
```

---

### Tâche 7 : campagne de bench + addendum au rapport

**Files:**
- Modify: `benchmarks/rapport-2026-07-03.md` (addendum 7)
- Résultats : `benchmarks/results/<date>/reference_results.json` + graphes

- [ ] **Step 1 : vérifier qu'Ollama et les modèles répondent**

```bash
curl -s http://localhost:11434/api/tags | .venv/bin/python -c \
  "import json,sys; print([m['name'] for m in json.load(sys.stdin)['models']])"
```

Attendu : `gpt-oss:120b` et `qwen3-coder-next:latest` dans la liste.

- [ ] **Step 2 : campagne (longue — lancer en arrière-plan)**

```bash
.venv/bin/python benchmarks/bench_reference.py --text all --pipeline all --runs 3
```

2 modèles × 2 textes × 2 pipelines × 3 runs = 24 runs (12 avec raffinement).
Ordre de grandeur ~2-3 h (120b ≈ 2-4 min/extraction sur ce volume, coder-next
~3× plus rapide, raffinement ≈ 2 appels de plus). Surveiller la première
itération pour attraper tôt une erreur de câblage.

- [ ] **Step 3 : inspecter les graphes, pas seulement les scores**

Pour chaque cellule du tableau, ouvrir au moins un graphe `ref_*.json` et
lire `false_positive_entities` / `missing_relations` / `failed_merges` du
`reference_results.json` : qualifier les faux positifs (hallucination vraie,
doublon de fusion, entité défendable oubliée par la référence ?). Si une
entité défendable manque à la référence → l'ajouter en `optional`, relancer
`.venv/bin/pytest tests/test_references.py -q`, puis
`.venv/bin/python benchmarks/rescore.py results/<date>` (pas de re-bench).

- [ ] **Step 4 : addendum 7 au rapport**

Ajouter à `benchmarks/rapport-2026-07-03.md` un « Addendum 7 (<date>) —
référence exhaustive : précision mesurée (axe 2) » avec :

- rappel du dispositif (2 textes inédits, référence à deux niveaux,
  appariement paires non orientées, spec en lien) ;
- tableau par texte : modèle × pipeline → P/R/F1 entités, P/R/F1 relations,
  fusions, temps (moyennes ± écart-type sur 3 runs) ;
- lecture : ce que la précision révèle que le rappel seul cachait
  (hallucinations ? doublons ? sur-extraction de la passe de complétude ?) ;
  comparaison nue vs actée — la complétude paie-t-elle sa précision ? ;
- limites : appariement prédicat-aveugle, deux textes courts d'un seul
  auteur (moi), niveaux core/optional = jugement d'annotateur unique.

- [ ] **Step 5 : commit + push**

```bash
git add benchmarks/rapport-2026-07-03.md benchmarks/results/
git commit -m "bench: campagne référence exhaustive — addendum 7 (P/R par modèle × pipeline)"
git push
```

(pousser aussi les commits des tâches 1-6 ; push direct sur main autorisé.)

- [ ] **Step 6 : mettre à jour la mémoire projet**

Mettre à jour `minerva-etat-et-prochaine-etape.md` (mémoire agent) : axe 2
réalisé (référence exhaustive + précision mesurée), pointer l'addendum 7 et
la spec ; retirer l'axe 2 de la liste des sujets ouverts.
