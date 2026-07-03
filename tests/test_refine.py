"""Les passes de raffinement doivent enrichir sans jamais dégrader : la
complétude n'ajoute que du nouveau (la fusion dédoublonne), et la
canonicalisation fusionne les coréférents sans perdre ni attributs ni
relations, en refusant les propositions incohérentes du LLM."""

from minerva.llm import ExtractedAttribute, ExtractedEntity, ExtractedRelation, ExtractionResult
from minerva.model import Entity, KnowledgeGraph, Relation
from minerva.refine import (
    CanonicalizationResult,
    MergeGroup,
    apply_canonicalization,
    canonicalize_graph,
    complete_graph,
)


class FakeBackend:
    def __init__(self, result):
        self._result = result
        self.prompts = []

    def parse(self, system, user, output_model):
        self.prompts.append(user)
        return self._result


def base_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Jean Valjean", type="personnage", attributes={"profession": "forçat"}))
    g.add_entity(Entity(name="Cosette", type="personnage"))
    g.add_relation(Relation(name="protège", source="Jean Valjean", target="Cosette"))
    return g


# --- Complétude ---------------------------------------------------------------

def test_completion_adds_missing_items_and_attributes():
    g = base_graph()
    backend = FakeBackend(ExtractionResult(
        entities=[
            ExtractedEntity(name="Javert", type="personnage"),
            # attribut manquant sur une entité existante
            ExtractedEntity(name="Jean Valjean", type="personnage",
                            attributes=[ExtractedAttribute(name="statut", value="maire")]),
        ],
        relations=[ExtractedRelation(name="poursuit", source="Javert", target="Jean Valjean")],
    ))

    complete_graph(g, "texte", backend)

    assert g.resolve("Javert") is not None
    valjean = g.resolve("Jean Valjean")
    assert valjean.attributes == {"profession": "forçat", "statut": "maire"}
    assert len(g.relations) == 2


def test_completion_redundant_output_changes_nothing():
    g = base_graph()
    before = g.to_dict()
    backend = FakeBackend(ExtractionResult(
        entities=[ExtractedEntity(name="Cosette", type="personnage")],
        relations=[ExtractedRelation(name="protège", source="Jean Valjean", target="Cosette")],
    ))

    complete_graph(g, "texte", backend)

    assert g.to_dict() == before


def test_completion_prompt_contains_graph_summary():
    g = base_graph()
    backend = FakeBackend(ExtractionResult())
    complete_graph(g, "le texte source", backend)

    prompt = backend.prompts[0]
    assert "le texte source" in prompt
    assert "Jean Valjean" in prompt and "--protège-->" in prompt


# --- Canonicalisation ----------------------------------------------------------

def coref_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_entity(Entity(name="évêque Myriel", type="personnage", attributes={"ville": "Digne"}))
    g.add_entity(Entity(name="Monseigneur Bienvenu", type="personnage", attributes={"vertu": "charité"}))
    g.add_entity(Entity(name="Jean Valjean", type="personnage"))
    g.add_relation(Relation(name="accueille", source="Monseigneur Bienvenu", target="Jean Valjean"))
    return g


def test_canonicalization_merges_group_keeping_attributes_and_relations():
    g = coref_graph()
    merged = apply_canonicalization(g, [
        MergeGroup(canonical="évêque Myriel", members=["évêque Myriel", "Monseigneur Bienvenu"]),
    ])

    assert len(merged.entities) == 2
    myriel = merged.resolve("Monseigneur Bienvenu")  # l'alias résout
    assert myriel is not None and myriel.name == "évêque Myriel"
    assert myriel.attributes == {"ville": "Digne", "vertu": "charité"}
    # la relation portée par l'ancien nom est re-câblée sur le canonique
    assert [(r.source, r.target) for r in merged.relations] == [("évêque Myriel", "Jean Valjean")]


def test_canonicalization_ignores_unknown_members_and_thin_groups():
    g = coref_graph()
    merged = apply_canonicalization(g, [
        MergeGroup(canonical="fantôme", members=["fantôme", "spectre"]),  # inconnus
        MergeGroup(canonical="Jean Valjean", members=["Jean Valjean"]),  # singleton
    ])
    assert merged.to_dict() == g.to_dict()


def test_canonicalization_member_claimed_once():
    g = coref_graph()
    merged = apply_canonicalization(g, [
        MergeGroup(canonical="évêque Myriel", members=["évêque Myriel", "Monseigneur Bienvenu"]),
        # tentative de re-fusionner Myriel ailleurs : ignorée
        MergeGroup(canonical="Jean Valjean", members=["Jean Valjean", "évêque Myriel"]),
    ])
    assert len(merged.entities) == 2
    assert merged.resolve("Jean Valjean").attributes == {}


def test_canonicalization_canonical_outside_members_falls_back():
    g = coref_graph()
    merged = apply_canonicalization(g, [
        MergeGroup(canonical="Jean Valjean",  # incohérent : pas du groupe
                   members=["évêque Myriel", "Monseigneur Bienvenu"]),
    ])
    # le premier membre valide devient canonique, Valjean n'est pas touché
    assert merged.resolve("évêque Myriel").name == "évêque Myriel"
    assert merged.resolve("Jean Valjean").attributes == {}
    assert len(merged.entities) == 2


def test_canonicalize_graph_end_to_end_with_fake_backend():
    g = coref_graph()
    backend = FakeBackend(CanonicalizationResult(groups=[
        MergeGroup(canonical="évêque Myriel", members=["évêque Myriel", "Monseigneur Bienvenu"]),
    ]))
    merged = canonicalize_graph(g, backend)

    assert "évêque Myriel" in backend.prompts[0] and "Monseigneur Bienvenu" in backend.prompts[0]
    assert len(merged.entities) == 2
