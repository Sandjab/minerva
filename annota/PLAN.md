# annota — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire `annota`, un atelier d'annotation d'entités local qui produit la vérité terrain de la fusion `canon_alias` de minerva et mesure l'écart (B³, LEA, paires sur-/sous-fusionnées).

**Architecture:** Projet Python séparé sous `annota/`, dépendant de `minerva` (import de `minerva.chunking`, lecture du schéma SQLite minerva). Cinq modules isolés — `metrics` (pur), `store` (gold.sqlite), `reader` (base minerva + re-découpage → candidats+contexte), `server` (HTTP stdlib), `web/index.html` (UI vanilla) — plus une CLI (`annota serve` / `annota score`).

**Tech Stack:** Python 3.10+ (bibliothèque standard uniquement : `sqlite3`, `http.server`, `json`), `pytest` en dev, HTML/CSS/JS vanilla sans build.

---

## File structure

```
annota/
  DESIGN.md            # existant
  PLAN.md              # ce document
  README.md            # usage (Task 7)
  pyproject.toml       # projet séparé, pytest en dev (Task 0)
  annota/
    __init__.py
    metrics.py         # Task 1-2 : bcubed, confusion_pairs, lea
    store.py           # Task 3 : gold.sqlite
    reader.py          # Task 4 : base minerva + re-chunk → candidats/contexte/partition
    server.py          # Task 5 : handlers purs + adaptateur http.server
    cli.py             # Task 5 : `annota serve` / `annota score`
  web/
    index.html         # Task 6 : UI (HTML/CSS/JS inline)
  tests/
    __init__.py
    conftest.py        # Task 4 : fixture base minerva synthétique
    test_metrics.py    # Task 1-2
    test_store.py      # Task 3
    test_reader.py     # Task 4
    test_server.py     # Task 5
```

**Clé de surface form partagée** (contrat entre `reader` et `store`) : un tuple
`(entity_id: int, kind: str, surface_form: str)` avec `kind ∈ {"name","alias"}`.
`reader` produit la partition prédite `P: {SurfaceKey → str(entity_id)}` ; `store`
produit la partition gold `G: {SurfaceKey → referent_id}`. `metrics` compare P et
G sur `P.keys() ∩ G.keys()`.

---

## Task 0 : Scaffolding du projet

**Files:**
- Create: `annota/pyproject.toml`
- Create: `annota/annota/__init__.py` (vide)
- Create: `annota/tests/__init__.py` (vide)
- Test: `annota/tests/test_env.py`

- [ ] **Step 1 : Écrire le test d'environnement**

```python
# annota/tests/test_env.py
def test_annota_importable():
    import annota  # noqa: F401

def test_minerva_chunking_reachable():
    from minerva.chunking import split_text, DEFAULT_CHUNK_SIZE
    assert DEFAULT_CHUNK_SIZE == 8000
    assert split_text("a\n\nb") == ["a\n\nb"]
```

- [ ] **Step 2 : Créer `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "annota"
version = "0.1.0"
description = "Atelier d'annotation d'entités & éval de fusion pour minerva"
requires-python = ">=3.10"
dependencies = []  # minerva fourni par le venv partagé (installé en -e)

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
annota = "annota.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["annota*"]
```

Créer aussi les fichiers vides `annota/annota/__init__.py` et `annota/tests/__init__.py`.

- [ ] **Step 3 : Installer et lancer**

Run (depuis `annota/`) : `pip install -e '.[dev]' && pytest tests/test_env.py -v`
Expected: 2 tests PASS (le venv contient déjà `minerva` en `-e`).

- [ ] **Step 4 : Commit**

```bash
git add annota/pyproject.toml annota/annota/__init__.py annota/tests/__init__.py annota/tests/test_env.py
git commit -m "feat(annota): scaffolding du projet + test d'environnement"
```

---

## Task 1 : metrics — B³ et décompte de paires

**Files:**
- Create: `annota/annota/metrics.py`
- Test: `annota/tests/test_metrics.py`

Chaque test encode *quel type d'erreur de fusion* la métrique doit détecter.

- [ ] **Step 1 : Écrire les tests B³ + paires**

```python
# annota/tests/test_metrics.py
import math
from annota.metrics import bcubed, confusion_pairs

def _almost(a, b): return math.isclose(a, b, abs_tol=1e-9)

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
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run : `pytest annota/tests/test_metrics.py -v`
Expected : FAIL (`ModuleNotFoundError: annota.metrics`).

- [ ] **Step 3 : Implémenter `bcubed` et `confusion_pairs`**

```python
# annota/annota/metrics.py
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
    # clusters restreints à l'univers
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
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run : `pytest annota/tests/test_metrics.py -v`
Expected : 4 tests PASS.

- [ ] **Step 5 : Commit**

```bash
git add annota/annota/metrics.py annota/tests/test_metrics.py
git commit -m "feat(annota): métriques B³ et décompte de paires sur-/sous-fusionnées"
```

---

## Task 2 : metrics — LEA *(différable : B³ + paires suffisent au premier signal)*

**Files:**
- Modify: `annota/annota/metrics.py`
- Test: `annota/tests/test_metrics.py` (ajout)

Convention singleton (Moosavi & Strube 2016) : `link(e) = |e|·(|e|−1)/2` pour
`|e| ≥ 2` ; pour un singleton, `link(e) = 1` et il n'est « résolu » que s'il est
aussi singleton dans l'autre partition.

- [ ] **Step 1 : Ajouter les tests LEA**

```python
# à ajouter dans annota/tests/test_metrics.py
from annota.metrics import lea

def test_lea_identical_is_one():
    pred = {"a": "1", "b": "1", "c": "2"}
    gold = {"a": "X", "b": "X", "c": "Y"}
    p, r, f = lea(pred, gold)
    assert _almost(p, 1.0) and _almost(r, 1.0) and _almost(f, 1.0)

def test_lea_over_merge_penalizes_precision():
    # pred fusionne {a,b,c} ; gold = {a,b} + {c}
    pred = {"a": "1", "b": "1", "c": "1"}
    gold = {"a": "X", "b": "X", "c": "Y"}
    # recall : gold {a,b} (link=1) parfaitement retrouvé dans pred → resolution=1 ;
    #   gold {c} singleton mais NON singleton dans pred → resolution=0.
    #   importance = taille : (2·1 + 1·0)/(2+1) = 2/3
    # precision : pred {a,b,c} (link=3) ; liens internes présents dans un même
    #   cluster gold : seul (a,b) → 1/3 ; importance 3 → (3·(1/3))/3 = 1/3
    p, r, f = lea(pred, gold)
    assert _almost(r, 2/3) and _almost(p, 1/3)
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run : `pytest annota/tests/test_metrics.py -k lea -v`
Expected : FAIL (`cannot import name 'lea'`).

- [ ] **Step 3 : Implémenter `lea`**

```python
# à ajouter dans annota/annota/metrics.py

def _link(size: int) -> float:
    return 1.0 if size == 1 else size * (size - 1) / 2.0


def _lea_directional(key: dict, resp: dict) -> float:
    """Score LEA directionnel (recall si key=gold,resp=pred ; precision si inversé)."""
    key_clusters: dict = {}
    for m, c in key.items():
        key_clusters.setdefault(c, []).append(m)
    resp_of: dict = resp  # m -> cluster resp
    num = den = 0.0
    for members in key_clusters.values():
        size = len(members)
        importance = size
        den += importance
        if size == 1:
            (m,) = members
            # singleton résolu ssi singleton aussi dans resp
            resp_c = resp_of[m]
            is_resp_singleton = sum(1 for x in resp_of if resp_of[x] == resp_c) == 1
            resolution = 1.0 if is_resp_singleton else 0.0
        else:
            # somme des liens internes retrouvés dans chaque cluster resp
            by_resp: dict = {}
            for m in members:
                by_resp.setdefault(resp_of[m], 0)
                by_resp[resp_of[m]] += 1
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
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run : `pytest annota/tests/test_metrics.py -v`
Expected : 6 tests PASS.

- [ ] **Step 5 : Commit**

```bash
git add annota/annota/metrics.py annota/tests/test_metrics.py
git commit -m "feat(annota): métrique LEA (convention singleton Moosavi & Strube)"
```

---

## Task 3 : store — vérité terrain (gold.sqlite)

**Files:**
- Create: `annota/annota/store.py`
- Test: `annota/tests/test_store.py`

- [ ] **Step 1 : Écrire les tests du store**

```python
# annota/tests/test_store.py
from annota.store import GoldStore

def test_create_and_roundtrip(tmp_path):
    path = tmp_path / "gold.sqlite"
    s = GoldStore.create(path, source_db="roman.sqlite")
    s.upsert(entity_id=1, kind="name", surface_form="Élise", referent_id="R1", referent_type="personnage")
    s.upsert(entity_id=2, kind="name", surface_form="Élise Blanchard", referent_id="R1")
    s.upsert(entity_id=3, kind="name", surface_form="11/03/2025", discarded=True)
    rows = {(r.entity_id, r.kind, r.surface_form): r for r in s.all()}
    assert rows[(1, "name", "Élise")].referent_id == "R1"
    assert rows[(3, "name", "11/03/2025")].discarded is True

def test_upsert_overwrites(tmp_path):
    s = GoldStore.create(tmp_path / "g.sqlite", source_db="x")
    s.upsert(entity_id=1, kind="name", surface_form="A", referent_id="R1")
    s.upsert(entity_id=1, kind="name", surface_form="A", referent_id="R2")  # correction
    rows = list(s.all())
    assert len(rows) == 1 and rows[0].referent_id == "R2"

def test_gold_partition_excludes_discards_and_undecided(tmp_path):
    s = GoldStore.create(tmp_path / "g.sqlite", source_db="x")
    s.upsert(entity_id=1, kind="name", surface_form="A", referent_id="R1")
    s.upsert(entity_id=2, kind="name", surface_form="B", referent_id="R1")
    s.upsert(entity_id=3, kind="name", surface_form="noise", discarded=True)
    s.upsert(entity_id=4, kind="name", surface_form="undecided")  # referent_id NULL
    part = s.gold_partition()
    assert part == {(1, "name", "A"): "R1", (2, "name", "B"): "R1"}
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run : `pytest annota/tests/test_store.py -v`
Expected : FAIL (`ModuleNotFoundError: annota.store`).

- [ ] **Step 3 : Implémenter `GoldStore`**

```python
# annota/annota/store.py
"""Vérité terrain d'annotation, persistée en SQLite pour écritures incrémentales."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS annotations (
    entity_id     INTEGER NOT NULL,
    kind          TEXT NOT NULL,            -- 'name' | 'alias'
    surface_form  TEXT NOT NULL,
    referent_id   TEXT,                     -- NULL = non décidé
    referent_type TEXT,
    discarded     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (entity_id, kind, surface_form)
);
"""


@dataclass
class Annotation:
    entity_id: int
    kind: str
    surface_form: str
    referent_id: str | None
    referent_type: str | None
    discarded: bool


class GoldStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @classmethod
    def create(cls, path, source_db: str) -> "GoldStore":
        conn = sqlite3.connect(str(path))
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('source_db', ?)", (source_db,))
        conn.commit()
        return cls(conn)

    @classmethod
    def open(cls, path) -> "GoldStore":
        return cls(sqlite3.connect(str(path)))

    def upsert(self, *, entity_id: int, kind: str, surface_form: str,
               referent_id: str | None = None, referent_type: str | None = None,
               discarded: bool = False) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO annotations "
            "(entity_id, kind, surface_form, referent_id, referent_type, discarded) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entity_id, kind, surface_form, referent_id, referent_type, int(discarded)),
        )
        self._conn.commit()

    def all(self) -> list[Annotation]:
        cur = self._conn.execute(
            "SELECT entity_id, kind, surface_form, referent_id, referent_type, discarded "
            "FROM annotations")
        return [Annotation(e, k, s, rid, rt, bool(d)) for e, k, s, rid, rt, d in cur]

    def gold_partition(self) -> dict:
        """{ (entity_id, kind, surface_form) -> referent_id } pour les surface
        forms décidées et non écartées."""
        return {
            (a.entity_id, a.kind, a.surface_form): a.referent_id
            for a in self.all()
            if a.referent_id is not None and not a.discarded
        }
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run : `pytest annota/tests/test_store.py -v`
Expected : 3 tests PASS.

- [ ] **Step 5 : Commit**

```bash
git add annota/annota/store.py annota/tests/test_store.py
git commit -m "feat(annota): gold store SQLite (upsert incrémental, partition gold)"
```

---

## Task 4 : reader — base minerva + contexte + partition prédite

**Files:**
- Create: `annota/annota/reader.py`
- Create: `annota/tests/conftest.py` (fixture base minerva synthétique)
- Test: `annota/tests/test_reader.py`

Le reader dépend du **schéma SQL de minerva** (contrat d'interface). La fixture
reproduit ce schéma minimal ; un test optionnel s'exécute sur `out/roman.sqlite`
s'il est présent.

- [ ] **Step 1 : Écrire la fixture de base minerva synthétique**

```python
# annota/tests/conftest.py
import sqlite3
import pytest

# Schéma minimal minerva utilisé par reader (contrat d'interface).
_MINERVA_SCHEMA = """
CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, type TEXT NOT NULL);
CREATE TABLE entity_aliases (entity_id INTEGER NOT NULL, alias TEXT NOT NULL, UNIQUE(entity_id, alias));
CREATE TABLE moments (id INTEGER PRIMARY KEY, chunk_index INTEGER NOT NULL, seq INTEGER NOT NULL, summary TEXT NOT NULL DEFAULT '');
CREATE TABLE appearances (moment_id INTEGER NOT NULL, entity_id INTEGER NOT NULL, UNIQUE(moment_id, entity_id));
CREATE TABLE assertions (id INTEGER PRIMARY KEY, entity_id INTEGER, relation_id INTEGER,
    attribute TEXT NOT NULL DEFAULT '', value TEXT NOT NULL DEFAULT '', moment_id INTEGER, chunk_index INTEGER);
CREATE VIEW entity_attributes AS
  SELECT a.entity_id, a.attribute AS name, a.value FROM assertions a
  WHERE a.entity_id IS NOT NULL AND a.attribute <> ''
    AND a.id = (SELECT MIN(b.id) FROM assertions b WHERE b.entity_id=a.entity_id AND b.attribute=a.attribute);
"""

@pytest.fixture
def minerva_db(tmp_path):
    """Base minerva synthétique : 2 entités (1 avec alias + attributs), assertions
    ancrées à des chunks, 1 moment résumé. Source = 2 chunks."""
    path = tmp_path / "mini.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_MINERVA_SCHEMA)
    conn.execute("INSERT INTO entities VALUES (1, 'Élise', 'personnage')")
    conn.execute("INSERT INTO entities VALUES (2, 'Anouck', 'inconnu')")
    conn.execute("INSERT INTO entity_aliases VALUES (1, 'Élise Blanchard')")
    conn.execute("INSERT INTO moments VALUES (10, 0, 0, 'Élise arrive.')")
    conn.execute("INSERT INTO appearances VALUES (10, 1)")
    # attribut d'Élise, ancré au chunk 0 ; mention d'Anouck au chunk 1
    conn.execute("INSERT INTO assertions VALUES (100, 1, NULL, 'rôle', 'protagoniste', 10, 0)")
    conn.execute("INSERT INTO assertions VALUES (101, 2, NULL, '', '', NULL, 1)")
    conn.commit()
    # source dont split_text(_, 8000) redonne exactement 2 chunks (2 paragraphes)
    source = tmp_path / "src.md"
    source.write_text("Chunk zéro : Élise arrive.\n\nChunk un : Anouck observe.", encoding="utf-8")
    return {"db": path, "source": source}
```

- [ ] **Step 2 : Écrire les tests du reader**

```python
# annota/tests/test_reader.py
import os
import sqlite3
import pytest
from annota.reader import surface_forms, predicted_partition, context_for_entity, build_chunks

def test_surface_forms_covers_names_and_aliases(minerva_db):
    conn = sqlite3.connect(minerva_db["db"])
    sfs = surface_forms(conn)
    keys = {(s.entity_id, s.kind, s.surface_form) for s in sfs}
    assert keys == {
        (1, "name", "Élise"),
        (2, "name", "Anouck"),
        (1, "alias", "Élise Blanchard"),
    }

def test_predicted_partition_groups_by_entity(minerva_db):
    conn = sqlite3.connect(minerva_db["db"])
    part = predicted_partition(conn)
    # les 2 surface forms de l'entité 1 partagent le cluster ; 2 clusters distincts
    assert part[(1, "name", "Élise")] == part[(1, "alias", "Élise Blanchard")]
    assert part[(1, "name", "Élise")] != part[(2, "name", "Anouck")]
    assert len(set(part.values())) == 2

def test_context_returns_attributes_and_passages(minerva_db):
    conn = sqlite3.connect(minerva_db["db"])
    chunks = build_chunks(minerva_db["source"].read_text(encoding="utf-8"))
    ctx = context_for_entity(conn, entity_id=1, chunks=chunks)
    assert ("rôle", "protagoniste") in [(a["name"], a["value"]) for a in ctx["attributes"]]
    assert any("Élise arrive" in p for p in ctx["passages"])   # chunk 0
    assert ctx["warnings"] == []

def test_context_warns_on_chunk_index_out_of_range(minerva_db):
    conn = sqlite3.connect(minerva_db["db"])
    ctx = context_for_entity(conn, entity_id=2, chunks=["un seul chunk"])  # chunk_index 1 hors bornes
    assert ctx["warnings"]  # avertissement de mismatch, pas d'exception

@pytest.mark.skipif(not os.path.exists("out/roman.sqlite"), reason="base réelle absente")
def test_real_base_has_421_surface_forms():
    conn = sqlite3.connect("out/roman.sqlite")
    assert len(surface_forms(conn)) == 421   # 370 names + 51 alias
```

- [ ] **Step 3 : Lancer, vérifier l'échec**

Run : `pytest annota/tests/test_reader.py -v`
Expected : FAIL (`ModuleNotFoundError: annota.reader`).

- [ ] **Step 4 : Implémenter `reader`**

```python
# annota/annota/reader.py
"""Lecture d'une base minerva → surface forms, partition prédite, contexte.

Dépend du schéma SQL de minerva et de minerva.chunking (même découpage qu'à
l'extraction, pour que assertions.chunk_index pointe sur le bon passage)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from minerva.chunking import DEFAULT_CHUNK_SIZE, split_text


@dataclass(frozen=True)
class SurfaceForm:
    entity_id: int
    kind: str          # 'name' | 'alias'
    surface_form: str
    entity_type: str


def surface_forms(conn: sqlite3.Connection) -> list[SurfaceForm]:
    out: list[SurfaceForm] = []
    for eid, name, etype in conn.execute("SELECT id, name, type FROM entities"):
        out.append(SurfaceForm(eid, "name", name, etype))
    for eid, alias, etype in conn.execute(
        "SELECT a.entity_id, a.alias, e.type FROM entity_aliases a "
        "JOIN entities e ON e.id = a.entity_id"
    ):
        out.append(SurfaceForm(eid, "alias", alias, etype))
    return out


def predicted_partition(conn: sqlite3.Connection) -> dict:
    """{ (entity_id, kind, surface_form) -> str(entity_id) } : toutes les surface
    forms d'une même entité partagent le cluster prédit."""
    return {
        (s.entity_id, s.kind, s.surface_form): str(s.entity_id)
        for s in surface_forms(conn)
    }


def build_chunks(source_text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    return split_text(source_text, chunk_size)


def context_for_entity(conn: sqlite3.Connection, entity_id: int, chunks: list[str]) -> dict:
    """Contexte d'une entité : attributs extraits, passages sources (via
    chunk_index), résumés de moments. `warnings` non vide si un chunk_index
    dépasse le nombre de chunks reconstitués (source/chunk_size incohérents)."""
    attributes = [
        {"name": name, "value": value}
        for name, value in conn.execute(
            "SELECT name, value FROM entity_attributes WHERE entity_id = ?", (entity_id,))
    ]
    chunk_indices = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT chunk_index FROM assertions "
            "WHERE entity_id = ? AND chunk_index IS NOT NULL ORDER BY chunk_index",
            (entity_id,))
    ]
    passages, warnings = [], []
    for i in chunk_indices:
        if 0 <= i < len(chunks):
            passages.append(chunks[i])
        else:
            warnings.append(f"chunk_index {i} hors bornes (source/chunk_size incohérents ?)")
    summaries = [
        row[0] for row in conn.execute(
            "SELECT m.summary FROM moments m JOIN appearances ap ON ap.moment_id = m.id "
            "WHERE ap.entity_id = ? AND m.summary <> ''", (entity_id,))
    ]
    return {"attributes": attributes, "passages": passages,
            "summaries": summaries, "warnings": warnings}
```

- [ ] **Step 5 : Lancer, vérifier le succès**

Run : `pytest annota/tests/test_reader.py -v`
Expected : 4 PASS + 1 SKIP (ou PASS si `out/roman.sqlite` présent).

- [ ] **Step 6 : Commit**

```bash
git add annota/annota/reader.py annota/tests/conftest.py annota/tests/test_reader.py
git commit -m "feat(annota): reader (surface forms, partition prédite, contexte par re-chunk)"
```

---

## Task 5 : server (handlers purs) + CLI

**Files:**
- Create: `annota/annota/server.py`
- Create: `annota/annota/cli.py`
- Test: `annota/tests/test_server.py`

La logique des routes est extraite en **fonctions pures testables** ; `http.server`
n'est qu'un adaptateur mince (non testé unitairement). Contrats JSON :

- `GET /api/candidates` → `{"candidates": [{entity_id, kind, surface_form,
  entity_type, predicted_cluster, context:{attributes,passages,summaries,warnings},
  annotation:{referent_id,referent_type,discarded}}...]}`
- `POST /api/annotate` (corps = une décision) → `{"ok": true}`
- `GET /api/score` → `{"bcubed":{p,r,f}, "lea":{p,r,f}, "over_merged":[...],
  "under_merged":[...], "n_evaluated":int, "n_discarded":int}`

- [ ] **Step 1 : Écrire les tests des handlers**

```python
# annota/tests/test_server.py
import sqlite3
from annota.server import build_candidates, apply_annotation, compute_score
from annota.store import GoldStore
from annota.reader import build_chunks

def test_build_candidates_merges_prediction_and_context(minerva_db, tmp_path):
    conn = sqlite3.connect(minerva_db["db"])
    store = GoldStore.create(tmp_path / "g.sqlite", source_db=str(minerva_db["db"]))
    chunks = build_chunks(minerva_db["source"].read_text(encoding="utf-8"))
    cands = build_candidates(conn, store, chunks)
    elise = next(c for c in cands if c["surface_form"] == "Élise")
    assert elise["predicted_cluster"] == "1"
    assert elise["annotation"]["referent_id"] is None
    assert any(a["name"] == "rôle" for a in elise["context"]["attributes"])

def test_apply_then_score(minerva_db, tmp_path):
    conn = sqlite3.connect(minerva_db["db"])
    store = GoldStore.create(tmp_path / "g.sqlite", source_db=str(minerva_db["db"]))
    # gold : Élise et son alias = même référent (correct) ; Anouck = autre référent
    apply_annotation(store, {"entity_id": 1, "kind": "name", "surface_form": "Élise", "referent_id": "R1"})
    apply_annotation(store, {"entity_id": 1, "kind": "alias", "surface_form": "Élise Blanchard", "referent_id": "R1"})
    apply_annotation(store, {"entity_id": 2, "kind": "name", "surface_form": "Anouck", "referent_id": "R2"})
    score = compute_score(conn, store)
    assert score["bcubed"]["f"] == 1.0     # prédiction == gold
    assert score["over_merged"] == [] and score["under_merged"] == []
    assert score["n_evaluated"] == 3
```

- [ ] **Step 2 : Lancer, vérifier l'échec**

Run : `pytest annota/tests/test_server.py -v`
Expected : FAIL (`ModuleNotFoundError: annota.server`).

- [ ] **Step 3 : Implémenter les handlers purs + l'adaptateur HTTP**

```python
# annota/annota/server.py
"""Handlers purs (testables) + adaptateur http.server mince pour l'atelier."""
from __future__ import annotations

import json
import sqlite3
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .metrics import bcubed, confusion_pairs, lea
from .reader import predicted_partition, surface_forms, context_for_entity
from .store import GoldStore

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def build_candidates(conn: sqlite3.Connection, store: GoldStore, chunks: list) -> list:
    ann = {(a.entity_id, a.kind, a.surface_form): a for a in store.all()}
    pred = predicted_partition(conn)
    ctx_cache: dict = {}
    out = []
    for s in surface_forms(conn):
        if s.entity_id not in ctx_cache:
            ctx_cache[s.entity_id] = context_for_entity(conn, s.entity_id, chunks)
        a = ann.get((s.entity_id, s.kind, s.surface_form))
        out.append({
            "entity_id": s.entity_id, "kind": s.kind, "surface_form": s.surface_form,
            "entity_type": s.entity_type,
            "predicted_cluster": pred[(s.entity_id, s.kind, s.surface_form)],
            "context": ctx_cache[s.entity_id],
            "annotation": {
                "referent_id": a.referent_id if a else None,
                "referent_type": a.referent_type if a else None,
                "discarded": a.discarded if a else False,
            },
        })
    return out


def apply_annotation(store: GoldStore, payload: dict) -> None:
    store.upsert(
        entity_id=payload["entity_id"], kind=payload["kind"],
        surface_form=payload["surface_form"],
        referent_id=payload.get("referent_id"),
        referent_type=payload.get("referent_type"),
        discarded=payload.get("discarded", False),
    )


def compute_score(conn: sqlite3.Connection, store: GoldStore) -> dict:
    pred = predicted_partition(conn)
    gold = store.gold_partition()
    bp, br, bf = bcubed(pred, gold)
    lp, lr, lf = lea(pred, gold)
    over, under = confusion_pairs(pred, gold)
    n_discarded = sum(1 for a in store.all() if a.discarded)
    return {
        "bcubed": {"p": bp, "r": br, "f": bf},
        "lea": {"p": lp, "r": lr, "f": lf},
        "over_merged": over, "under_merged": under,
        "n_evaluated": len([k for k in pred if k in gold]),
        "n_discarded": n_discarded,
    }


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, conn, store, chunks, **kwargs):
        self._conn, self._store, self._chunks = conn, store, chunks
        super().__init__(*args, **kwargs)

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = (WEB_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/candidates":
            self._send_json({"candidates": build_candidates(self._conn, self._store, self._chunks)})
        elif self.path == "/api/score":
            self._send_json(compute_score(self._conn, self._store))
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            apply_annotation(self._store, payload)
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)


def serve(conn, store, chunks, host="127.0.0.1", port=8000):
    handler = partial(_Handler, conn=conn, store=store, chunks=chunks)
    httpd = HTTPServer((host, port), handler)
    print(f"annota sur http://{host}:{port}")
    httpd.serve_forever()
```

```python
# annota/annota/cli.py
"""CLI annota : `annota serve` (atelier) et `annota score` (métriques)."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .reader import build_chunks
from .server import compute_score, serve
from .store import GoldStore


def _open(base: str, gold: str, source: str | None, chunk_size: int):
    conn = sqlite3.connect(base)
    gpath = Path(gold)
    store = GoldStore.open(gpath) if gpath.exists() else GoldStore.create(gpath, source_db=base)
    chunks = build_chunks(Path(source).read_text(encoding="utf-8"), chunk_size) if source else []
    return conn, store, chunks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="annota")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="lancer l'atelier d'annotation")
    ps.add_argument("base"); ps.add_argument("--source", required=True)
    ps.add_argument("--gold", default="gold.sqlite")
    ps.add_argument("--chunk-size", type=int, default=8000)
    ps.add_argument("--port", type=int, default=8000)

    pc = sub.add_parser("score", help="mesurer canon_alias contre le gold")
    pc.add_argument("base"); pc.add_argument("--gold", default="gold.sqlite")

    args = p.parse_args(argv)
    if args.cmd == "serve":
        conn, store, chunks = _open(args.base, args.gold, args.source, args.chunk_size)
        serve(conn, store, chunks, port=args.port)
        return 0
    if args.cmd == "score":
        conn = sqlite3.connect(args.base)
        store = GoldStore.open(Path(args.gold))
        print(json.dumps(compute_score(conn, store), ensure_ascii=False, indent=2))
        return 0
    return 1
```

- [ ] **Step 4 : Lancer, vérifier le succès**

Run : `pytest annota/tests/test_server.py -v`
Expected : 2 tests PASS.

- [ ] **Step 5 : Commit**

```bash
git add annota/annota/server.py annota/annota/cli.py annota/tests/test_server.py
git commit -m "feat(annota): handlers d'annotation/score + serveur http + CLI serve/score"
```

---

## Task 6 : web/index.html — UI d'annotation

**Files:**
- Create: `annota/web/index.html`

Page unique, HTML/CSS/JS vanilla inline. Testée **manuellement** (une UI ne se
teste pas unitairement sans navigateur ; le harnais back-end est déjà couvert).

Spécification de comportement (à implémenter dans `index.html`) :

- **Layout deux colonnes** : gauche = liste des candidats (surface form, type
  prédit, badge d'état d'annotation) ; droite = panneau contexte du candidat
  sélectionné (attributs en table, passages sources, résumés, + bandeau si
  `context.warnings` non vide).
- **Au chargement** : `fetch('/api/candidates')` puis rendu de la liste. Filtre
  rapide : « seulement les clusters à alias » (candidats dont l'entité a ≥1 alias)
  et « non annotés ».
- **Sélection d'un candidat** : affiche son contexte à droite + les contrôles :
  - champ **referent_id** (texte libre ; autocomplétion sur les referent_id déjà
    saisis pour regrouper) ;
  - champ **type** (personnage/lieu/objet/organisation/autre) ;
  - bouton **Écarter** (discarded=true) ; bouton **Valider**.
- **Enregistrement** : chaque validation `POST /api/annotate` avec le corps
  `{entity_id, kind, surface_form, referent_id?, referent_type?, discarded?}`,
  puis met à jour le badge d'état du candidat sans recharger toute la liste.
- **Barre de progression** : `n annotés / n total` ; bouton **Score** →
  `fetch('/api/score')` et affiche B³/LEA, `n_evaluated`, `n_discarded`, et les
  premières paires `over_merged` / `under_merged` (exemples à inspecter).

Squelette JS minimal (les appels réseau ; le rendu DOM est à compléter) :

```html
<script>
async function loadCandidates() {
  const r = await fetch('/api/candidates');
  return (await r.json()).candidates;
}
async function annotate(decision) {
  await fetch('/api/annotate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(decision)});
}
async function loadScore() {
  const r = await fetch('/api/score');
  return await r.json();
}
</script>
```

- [ ] **Step 1 : Écrire `annota/web/index.html`** (layout + rendu + les 3 fetch ci-dessus)

- [ ] **Step 2 : Vérification manuelle**

```bash
cd annota && annota serve ../out/roman.sqlite --source ../in/roman.md --gold ../out/roman.gold.sqlite --port 8000
```
Checklist dans le navigateur (http://127.0.0.1:8000) :
- la liste se charge ; le filtre « clusters à alias » réduit à ~23 entités ;
- sélectionner un candidat affiche attributs + au moins un passage source ;
- saisir un referent_id + Valider met à jour le badge ; recharger la page conserve l'annotation (persistée dans le gold) ;
- **Score** affiche B³/LEA et des exemples de paires.

- [ ] **Step 3 : Commit**

```bash
git add annota/web/index.html
git commit -m "feat(annota): UI d'annotation (liste + contexte + score)"
```

---

## Task 7 : README

**Files:**
- Create: `annota/README.md`

- [ ] **Step 1 : Écrire le README**

Contenu : but de l'outil ; prérequis (venv avec `minerva` installé, `pip install -e '.[dev]'`) ; les deux commandes avec un exemple réel :

```
# annoter (atelier web) — base minerva + roman source + gold de sortie
annota serve out/roman.sqlite --source in/roman.md --gold out/roman.gold.sqlite

# mesurer canon_alias contre le gold annoté
annota score out/roman.sqlite --gold out/roman.gold.sqlite
```

Rappeler : `--chunk-size` doit valoir celui utilisé à l'extraction (défaut 8000) ;
commencer par le filtre « clusters à alias » (sur-fusion) ; lien vers `DESIGN.md`.

- [ ] **Step 2 : Commit**

```bash
git add annota/README.md
git commit -m "docs(annota): README (usage serve/score)"
```

---

## Notes d'exécution

- **Où lancer pytest** : depuis `annota/` (`cd annota && pytest -v`). Le test réel
  `test_real_base_has_421_surface_forms` ne s'exécute que si `out/roman.sqlite`
  existe (chemin relatif au cwd) — sinon SKIP.
- **Ordre** : les tâches sont indépendantes par fichier mais s'enchaînent par
  dépendance logique (metrics → store → reader → server → web → readme).
- **LEA (Task 2) différable** : si le premier signal presse, sauter Task 2 ;
  `compute_score` appelle `lea` — dans ce cas, remplacer temporairement son
  résultat par `{"p":0,"r":0,"f":0}` n'est pas nécessaire car Task 5 vient après
  Task 2 dans l'ordre. Ne sauter Task 2 que si l'on réordonne sciemment.
```
