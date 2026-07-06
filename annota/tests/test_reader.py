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


@pytest.mark.skipif(not os.path.exists("out/roman.sqlite"), reason="base réelle absente")
def test_real_base_has_421_surface_forms():
    conn = sqlite3.connect("out/roman.sqlite")
    assert len(surface_forms(conn)) == 421   # 370 names + 51 alias
