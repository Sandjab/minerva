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
    resolve_aliases,
)


class FakeBackend:
    def __init__(self, result):
        self._result = result
        self.prompts = []
        self.systems = []

    def parse(self, system, user, output_model):
        self.systems.append(system)
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


def test_canonicalisation_preserve_le_journal_temporel():
    from minerva.model import Assertion
    from minerva.refine import MergeGroup, apply_canonicalization

    graph = KnowledgeGraph()
    m = graph.timeline.add_moment(0, 0, "à Digne")
    graph.add_entity(Entity(name="évêque Myriel", type="personnage"))
    graph.add_entity(Entity(name="Monseigneur Bienvenu", type="personnage"))
    graph.timeline.add_appearance(m.id, "évêque Myriel")
    graph.add_assertion(
        Assertion(entity="évêque Myriel", attribute="ville", value="Digne", moment_id=m.id)
    )

    merged = apply_canonicalization(
        graph,
        [MergeGroup(canonical="Monseigneur Bienvenu",
                    members=["Monseigneur Bienvenu", "évêque Myriel"])],
    )

    assert len(merged.entities) == 1
    [a] = merged.assertions
    assert a.entity == "Monseigneur Bienvenu"
    assert a.moment_id == m.id
    assert merged.timeline.appearances == {m.id: {"Monseigneur Bienvenu"}}
    assert merged.resolve("Monseigneur Bienvenu").attributes == {"ville": "Digne"}


# --- Passe d'alias / identité d'emprunt (relit le TEXTE) ----------------------

def impersonation_graph() -> KnowledgeGraph:
    """Deux entités que la seule liste de noms ne peut pas rapprocher :
    l'identité d'emprunt n'est révélée que par le texte."""
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Antoine Sérac", type="personne",
                        attributes={"métier": "cartographe"}))
    g.add_entity(Entity(name="Théo Rivière", type="personne"))
    g.add_relation(Relation(name="exerce à", source="Antoine Sérac", target="Valsonne"))
    return g


def test_resolve_aliases_merges_impersonation_revealed_by_text():
    g = impersonation_graph()
    backend = FakeBackend(CanonicalizationResult(groups=[
        MergeGroup(canonical="Antoine Sérac", members=["Antoine Sérac", "Théo Rivière"]),
    ]))

    merged = resolve_aliases(g, "… il s'appelait Théo Rivière …", backend)

    # les deux noms résolvent désormais vers UNE SEULE entité (fusion faite)
    a = merged.resolve("Antoine Sérac")
    b = merged.resolve("Théo Rivière")
    assert a is not None and b is not None and a is b
    assert a.name == "Antoine Sérac"
    # l'attribut porté par l'entité fusionnée est conservé
    assert a.attributes == {"métier": "cartographe"}


def test_resolve_aliases_prompt_carries_text_and_entities():
    g = impersonation_graph()
    backend = FakeBackend(CanonicalizationResult())

    resolve_aliases(g, "révélation clef : Théo Rivière", backend)

    prompt = backend.prompts[0]
    # le texte source est fourni — c'est LA différence avec la canonicalisation
    assert "révélation clef : Théo Rivière" in prompt
    # la liste des entités est fournie aussi
    assert "Antoine Sérac" in prompt and "Théo Rivière" in prompt


def test_resolve_aliases_scope_selects_distinct_system_prompt():
    """Les deux portées (ciblée / large) doivent réellement différer, sinon le
    bench comparerait deux fois le même prompt."""
    g = impersonation_graph()
    b_cible = FakeBackend(CanonicalizationResult())
    b_large = FakeBackend(CanonicalizationResult())

    resolve_aliases(g, "texte", b_cible, scope="impersonation")
    resolve_aliases(g, "texte", b_large, scope="broad")

    assert b_cible.systems[0] and b_large.systems[0]
    assert b_cible.systems[0] != b_large.systems[0]


def test_resolve_aliases_windows_long_text_carrying_global_entities():
    """Passe alias scalable : sur un texte plus long que la fenêtre, un appel
    par fenêtre, chacun portant la liste GLOBALE des entités (pour relier des
    mentions dispersées sur un roman sans mettre tout le texte dans un prompt)."""
    g = impersonation_graph()
    backend = FakeBackend(CanonicalizationResult())

    resolve_aliases(g, "Alpha.\n\nBeta.\n\nGamma.", backend, window_size=8)

    assert len(backend.prompts) == 3  # une fenêtre par paragraphe
    for prompt in backend.prompts:
        assert "Antoine Sérac" in prompt  # liste globale fournie à chaque fenêtre
    # chaque fenêtre ne porte que son fragment de texte
    assert "Beta." in backend.prompts[1] and "Alpha." not in backend.prompts[1]


def test_resolve_aliases_single_window_unchanged():
    """Non-régression : un texte qui tient dans la fenêtre = un seul appel
    (comportement d'avant le fenêtrage)."""
    g = impersonation_graph()
    backend = FakeBackend(CanonicalizationResult(groups=[
        MergeGroup(canonical="Antoine Sérac", members=["Antoine Sérac", "Théo Rivière"]),
    ]))
    merged = resolve_aliases(g, "… il s'appelait Théo Rivière …", backend)
    assert len(backend.prompts) == 1
    assert merged.resolve("Antoine Sérac") is merged.resolve("Théo Rivière")
