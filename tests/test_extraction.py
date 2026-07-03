"""L'orchestration doit fusionner les résultats de tous les chunks et
transmettre les entités déjà connues au chunk suivant — c'est le mécanisme qui
évite les doublons d'entités entre chapitres."""

from minerva.extraction import build_user_prompt, extract_graph
from minerva.llm import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)


class FakeBackend:
    """Renvoie un résultat préparé par appel et enregistre les prompts reçus."""

    def __init__(self, results):
        self._results = list(results)
        self.prompts = []

    def parse(self, system, user, output_model):
        self.prompts.append(user)
        return self._results.pop(0)


def test_results_from_all_chunks_are_merged():
    chunk1 = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Jean Valjean",
                type="personnage",
                attributes=[ExtractedAttribute(name="profession", value="forçat")],
            )
        ]
    )
    chunk2 = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Jean Valjean",
                type="personnage",
                attributes=[ExtractedAttribute(name="statut", value="maire")],
            ),
            ExtractedEntity(name="Cosette", type="personnage"),
        ],
        relations=[
            ExtractedRelation(name="protège", source="Jean Valjean", target="Cosette")
        ],
    )
    backend = FakeBackend([chunk1, chunk2])
    text = "Premier paragraphe.\n\nSecond paragraphe."

    graph = extract_graph(text, backend, chunk_size=20)

    assert len(backend.prompts) == 2
    valjean = graph.resolve("Jean Valjean")
    assert valjean is not None
    assert valjean.attributes == {"profession": "forçat", "statut": "maire"}
    assert len(graph.relations) == 1


def test_known_entities_are_passed_to_next_chunk():
    chunk1 = ExtractionResult(entities=[ExtractedEntity(name="Cosette", type="personnage")])
    chunk2 = ExtractionResult()
    backend = FakeBackend([chunk1, chunk2])

    extract_graph("Un.\n\nDeux.", backend, chunk_size=5)

    assert "Cosette" not in backend.prompts[0]
    assert "Cosette" in backend.prompts[1]


def test_progress_callback_reports_each_chunk():
    backend = FakeBackend([ExtractionResult(), ExtractionResult()])
    calls = []

    extract_graph("Un.\n\nDeux.", backend, chunk_size=5, on_progress=lambda d, t: calls.append((d, t)))

    assert calls == [(1, 2), (2, 2)]


def test_user_prompt_caps_known_entities():
    known = [f"Entité {i}" for i in range(500)]
    prompt = build_user_prompt("texte", known)
    assert "Entité 199" in prompt
    assert "Entité 200" not in prompt
