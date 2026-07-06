import os
import sqlite3

import pytest

from annota.reader import (build_chunks, context_for_entity, predicted_partition,
                           surface_forms)


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


def test_context_windows_around_mention_not_full_chunk(minerva_db):
    # un chunk long : le passage doit être une fenêtre autour de la mention,
    # pas le chunk entier (sinon illisible pour les personnages principaux).
    conn = sqlite3.connect(minerva_db["db"])
    long_chunk = "a" * 300 + " Élise arrive ici. " + "b" * 300
    ctx = context_for_entity(conn, entity_id=1, chunks=[long_chunk], window=30)
    assert ctx["passages"], "au moins une fenêtre attendue"
    assert all("Élise" in p for p in ctx["passages"])
    assert all(len(p) < len(long_chunk) for p in ctx["passages"]), "fenêtré, pas le chunk entier"
    assert any(p.startswith("…") or p.endswith("…") for p in ctx["passages"]), "bords tronqués"


def test_context_caps_and_exposes_totals(minerva_db):
    conn = sqlite3.connect(minerva_db["db"])
    chunks = build_chunks(minerva_db["source"].read_text(encoding="utf-8"))
    ctx = context_for_entity(conn, entity_id=1, chunks=chunks, max_attributes=1)
    assert len(ctx["attributes"]) <= 1
    assert ctx["n_attributes_total"] >= len(ctx["attributes"])
    assert "n_passages_total" in ctx


@pytest.mark.skipif(not os.path.exists("out/roman.sqlite"), reason="base réelle absente")
def test_real_base_has_421_surface_forms():
    conn = sqlite3.connect("out/roman.sqlite")
    assert len(surface_forms(conn)) == 421   # 370 names + 51 alias
