"""Les passes de raffinement doivent enrichir sans jamais dégrader : la
complétude n'ajoute que du nouveau (la fusion dédoublonne), et la
canonicalisation fusionne les coréférents sans perdre ni attributs ni
relations, en refusant les propositions incohérentes du LLM."""

import pytest
from pydantic import ValidationError

from minerva.llm import ExtractedAttribute, ExtractedEntity, ExtractedRelation, ExtractionResult
from minerva.model import Assertion, Entity, KnowledgeGraph, Relation
from minerva.refine import (
    ENTITY_TYPES,
    CanonicalizationResult,
    EntityType,
    MergeGroup,
    TypingResult,
    apply_canonicalization,
    canonicalize_graph,
    complete_graph,
    refine_graph,
    resolve_aliases,
    type_entities,
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


# --- Passe de typage des entités « inconnu » ----------------------------------

def untyped_graph() -> KnowledgeGraph:
    """« carnet noir » n'apparaît que comme extrémité de relation et sujet de
    fait : il est créé « inconnu », jamais déclaré comme entité typée — le cas
    qui domine le taux d'inconnu à l'extraction."""
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Élise", type="personnage"))
    g.add_relation(Relation(name="écrit dans", source="Élise", target="carnet noir"))
    g.add_assertion(Assertion(entity="carnet noir", attribute="couleur", value="noir"))
    return g


def test_type_entities_assigns_type_to_inconnu():
    g = untyped_graph()
    carnet = g.resolve("carnet noir")
    assert carnet is not None and carnet.type == "inconnu"
    backend = FakeBackend(TypingResult(types=[EntityType(name="carnet noir", type="objet")]))

    n = type_entities(g, backend)

    assert n == 1
    typed = g.resolve("carnet noir")
    assert typed is not None and typed.type == "objet"


def test_type_entities_ignores_real_types_and_unknown_names():
    """La passe ne touche QUE les « inconnu » : elle n'écrase pas un vrai type
    et ignore un nom qu'elle ne connaît pas."""
    g = untyped_graph()
    backend = FakeBackend(TypingResult(types=[
        EntityType(name="Élise", type="objet"),        # déjà typée -> ignorée
        EntityType(name="fantôme", type="objet"),      # inconnue du graphe -> ignorée
        EntityType(name="carnet noir", type="objet"),  # seule application valide
    ]))

    n = type_entities(g, backend)

    assert n == 1
    elise = g.resolve("Élise")
    assert elise is not None and elise.type == "personnage"


def test_type_entities_prompt_lists_inconnu_with_context():
    g = untyped_graph()
    backend = FakeBackend(TypingResult())

    type_entities(g, backend)

    prompt = backend.prompts[0]
    assert "carnet noir" in prompt            # l'entité à typer
    assert "couleur=noir" in prompt           # son contexte d'attribut
    assert "écrit dans" in prompt             # son contexte relationnel
    assert "Élise" not in prompt.split("carnet noir")[0]  # les entités déjà typées ne sont pas à typer


def test_entity_type_vocabulary_is_closed():
    """Le type est contraint à la liste fermée (via `enum` du json_schema) :
    un type hors-liste est rejeté, ce qui normalise les étiquettes (plus de
    « personnages » vs « personnage », plus de « boisson »/« meuble » épars)."""
    EntityType(name="carnet noir", type="objet")  # dans la liste : OK
    with pytest.raises(ValidationError):
        EntityType.model_validate({"name": "carnet noir", "type": "boisson"})  # hors liste : rejeté
    assert "personnage" in ENTITY_TYPES and "autre" in ENTITY_TYPES
    assert "boisson" not in ENTITY_TYPES


def test_typing_prompt_lists_the_closed_vocabulary():
    g = untyped_graph()
    backend = FakeBackend(TypingResult())
    type_entities(g, backend)
    system = backend.systems[0]
    for allowed in ENTITY_TYPES:
        assert allowed in system  # la liste fermée est fournie au modèle


def test_type_entities_no_inconnu_skips_backend():
    """Rien à typer -> aucun appel LLM (économie sur les graphes déjà propres)."""
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Élise", type="personnage"))
    backend = FakeBackend(TypingResult())

    assert type_entities(g, backend) == 0
    assert backend.prompts == []


class TypingByPromptBackend:
    """Backend factice qui type « objet » CHAQUE entité listée dans le prompt
    (une par ligne « - nom »). Contrairement à `FakeBackend` (réponse fixe), il
    répond selon le lot reçu : indispensable pour vérifier que le fenêtrage
    applique les types de TOUS les lots, pas seulement du premier."""

    def __init__(self):
        self.prompts = []

    def parse(self, system, user, output_model):
        self.prompts.append(user)
        names = [line[2:] for line in user.splitlines() if line.startswith("- ")]
        return TypingResult(types=[EntityType(name=n, type="objet") for n in names])


def test_type_entities_windows_many_inconnu_into_batches():
    """Passe de typage scalable : sur plus d'entités « inconnu » que la taille de
    lot, un appel LLM par lot (chacun ne portant que son sous-ensemble), et les
    types de TOUS les lots sont appliqués — sinon typer un roman entier en un
    seul prompt dépasse le contexte et perd des entités."""
    g = KnowledgeGraph()
    for i in range(5):
        g.add_entity(Entity(name=f"objet{i}", type="inconnu"))
    backend = TypingByPromptBackend()

    n = type_entities(g, backend, batch_size=2)

    assert len(backend.prompts) == 3  # ceil(5/2) : lots de 2, 2, 1
    assert n == 5  # toutes les entités typées, à travers les lots
    for i in range(5):
        assert g.resolve(f"objet{i}").type == "objet"
    # chaque lot ne porte que son sous-ensemble d'entités
    assert backend.prompts[0].count("\n- objet") == 2
    assert backend.prompts[2].count("\n- objet") == 1


def test_type_entities_single_batch_unchanged():
    """Non-régression : moins d'entités que la taille de lot = un seul appel
    (comportement d'avant le fenêtrage)."""
    g = untyped_graph()
    backend = FakeBackend(TypingResult(types=[EntityType(name="carnet noir", type="objet")]))

    n = type_entities(g, backend)

    assert len(backend.prompts) == 1
    assert n == 1
    assert g.resolve("carnet noir").type == "objet"


class TypesFirstOnlyBackend:
    """Backend « paresseux » : ne type que la PREMIÈRE entité listée dans chaque
    prompt (simule un LLM qui en oublie à chaque appel — la cause du résiduel
    observé à l'échelle roman). La boucle de convergence doit rattraper les
    oubliés en re-soumettant les « inconnu » restants dans de nouveaux lots."""

    def __init__(self):
        self.calls = 0

    def parse(self, system, user, output_model):
        self.calls += 1
        names = [line[2:] for line in user.splitlines() if line.startswith("- ")]
        return TypingResult(types=[EntityType(name=n, type="objet") for n in names[:1]])


def test_type_entities_loops_until_all_typed():
    """Boucle de convergence : un LLM qui oublie des entités par appel est rattrapé
    en re-typant les « inconnu » restants jusqu'à ce qu'un passage ne progresse
    plus. Sans boucle, seule la 1re entité serait typée."""
    g = KnowledgeGraph()
    for name in ("a", "b", "c"):
        g.add_entity(Entity(name=name, type="inconnu"))
    backend = TypesFirstOnlyBackend()  # ne type qu'UNE entité par appel

    n = type_entities(g, backend)

    assert n == 3  # les trois typées, à travers plusieurs passages
    assert all(g.resolve(x).type == "objet" for x in ("a", "b", "c"))
    assert backend.calls == 3  # un passage par entité récupérée, puis arrêt (0 inconnu)


def test_type_entities_stops_when_no_progress():
    """Terminaison : si un passage ne type plus rien (restants irrécupérables —
    bruit d'extraction, noms non résolus), la boucle s'arrête au lieu de tourner
    indéfiniment."""
    g = KnowledgeGraph()
    g.add_entity(Entity(name="(trajet — pas une entité)", type="inconnu"))
    backend = FakeBackend(TypingResult())  # ne type jamais rien : aucun progrès

    n = type_entities(g, backend)

    assert n == 0
    assert g.resolve("(trajet — pas une entité)").type == "inconnu"
    assert len(backend.prompts) == 1  # un seul passage, puis arrêt faute de progrès


# --- Orchestration du raffinement ---------------------------------------------

class DispatchBackend:
    """Backend factice qui répond selon le type de sortie demandé (les passes de
    raffinement n'utilisent pas le même modèle de résultat)."""

    def __init__(self, by_output):
        self._by = by_output

    def parse(self, system, user, output_model):
        return self._by.get(output_model, output_model())


def test_refine_graph_enchaine_fusion_puis_typage():
    """`refine_graph` = canonicalisation -> alias -> typage : après lui, une
    entité restée « inconnu » à l'extraction est typée."""
    g = untyped_graph()  # Élise (personnage) + « carnet noir » (inconnu)
    backend = DispatchBackend({
        CanonicalizationResult: CanonicalizationResult(),  # aucune fusion proposée
        TypingResult: TypingResult(types=[EntityType(name="carnet noir", type="objet")]),
    })

    refined = refine_graph(g, "texte source", backend)

    carnet = refined.resolve("carnet noir")
    assert carnet is not None and carnet.type == "objet"
    elise = refined.resolve("Élise")
    assert elise is not None and elise.type == "personnage"  # inchangée
