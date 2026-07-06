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
