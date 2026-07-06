import sqlite3

from annota.reader import build_chunks
from annota.server import apply_annotation, build_candidates, compute_score
from annota.store import GoldStore


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
