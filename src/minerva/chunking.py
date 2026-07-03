"""Découpage d'un texte en chunks aux frontières de paragraphes."""

from __future__ import annotations

DEFAULT_CHUNK_SIZE = 8_000  # caractères


def split_text(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Découpe `text` en chunks d'au plus `max_chars` caractères.

    Les paragraphes (séparés par des lignes vides) sont préservés ; un
    paragraphe plus long que `max_chars` est découpé brutalement.
    """
    if max_chars <= 0:
        raise ValueError("max_chars doit être positif")

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        # +2 pour le séparateur "\n\n"
        if current_len + len(para) + (2 if current else 0) > max_chars:
            flush()
        current.append(para)
        current_len += len(para) + (2 if current_len else 0)

    flush()
    return chunks
