def test_annota_importable():
    import annota  # noqa: F401


def test_minerva_chunking_reachable():
    from minerva.chunking import split_text, DEFAULT_CHUNK_SIZE
    assert DEFAULT_CHUNK_SIZE == 8000
    assert split_text("a\n\nb") == ["a\n\nb"]
