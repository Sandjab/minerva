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
    "reference_fusion_texte.json": "fusion_texte.txt",
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
