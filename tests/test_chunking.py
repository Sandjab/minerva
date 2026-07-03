"""Le chunking doit couvrir tout le texte sans dépasser la taille maximale,
sinon des passages du roman échapperaient à l'extraction ou déborderaient le
contexte des modèles locaux."""

import pytest

from minerva.chunking import split_text


def test_short_text_is_one_chunk():
    assert split_text("Un court paragraphe.", max_chars=100) == ["Un court paragraphe."]


def test_paragraphs_grouped_without_exceeding_limit():
    paras = [f"Paragraphe {i} " + "x" * 40 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = split_text(text, max_chars=120)

    assert all(len(c) <= 120 for c in chunks)
    # aucun paragraphe perdu
    assert "".join(chunks).replace("\n\n", "") == text.replace("\n\n", "")


def test_oversized_paragraph_is_hard_split():
    text = "a" * 250
    chunks = split_text(text, max_chars=100)
    assert chunks == ["a" * 100, "a" * 100, "a" * 50]


def test_empty_text_yields_no_chunk():
    assert split_text("   \n\n  ", max_chars=100) == []


def test_invalid_max_chars_rejected():
    with pytest.raises(ValueError):
        split_text("texte", max_chars=0)
