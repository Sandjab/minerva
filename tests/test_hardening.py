"""Durcissements issus du bench du 2026-07-03 : les LLM produisent des tirets
Unicode exotiques, des préfixes de titre incohérents et des enregistrements
vides — le pipeline doit y résister sans fausse fusion."""

import pytest

from minerva.extraction import sanitize
from minerva.llm import ExtractedEntity, ExtractedRelation, ExtractionResult
from minerva.model import Entity, KnowledgeGraph, Relation, normalize, strip_title


# --- 1. Normalisation Unicode ------------------------------------------------

def test_normalize_unifies_typographic_dashes_and_apostrophes():
    # U+2011 (tiret insécable) observé dans la sortie de gpt-oss:120b
    assert normalize("Montreuil‑sur‑Mer") == normalize("Montreuil-sur-Mer")
    assert normalize("l’évêque") == normalize("l'évêque")


def test_entities_differing_only_by_dash_variant_are_merged():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Montreuil-sur-Mer", type="lieu"))
    g.add_entity(Entity(name="Montreuil‑sur‑Mer", type="lieu", attributes={"région": "Pas-de-Calais"}))

    assert len(g.entities) == 1
    e = g.resolve("Montreuil-sur-Mer")
    assert e is not None and e.attributes == {"région": "Pas-de-Calais"}


# --- 2. Repli titres/articles ------------------------------------------------

def test_strip_title_removes_article_then_title():
    assert strip_title(normalize("l'inspecteur Javert")) == "javert"
    assert strip_title(normalize("M. Madeleine")) == "madeleine"
    assert strip_title(normalize("les Thénardier")) == "thénardier"
    assert strip_title("javert") is None  # inchangé -> pas de repli


def test_titled_and_bare_names_resolve_to_same_entity():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Javert", type="personnage"))
    merged = g.add_entity(Entity(name="inspecteur Javert", type="personnage", attributes={"métier": "policier"}))

    assert len(g.entities) == 1
    assert merged.name == "Javert"
    assert g.resolve("l'inspecteur Javert") is g.resolve("Javert")


def test_bare_name_resolves_to_titled_entity_added_first():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="inspecteur Javert", type="personnage"))

    assert g.resolve("Javert") is not None


def test_ambiguous_stripped_form_disables_fallback():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="M. Thénardier", type="personnage"))
    g.add_entity(Entity(name="Mme Thénardier", type="personnage"))

    assert len(g.entities) == 2  # pas de fausse fusion des époux
    assert g.resolve("Thénardier") is None  # forme ambiguë : repli désactivé


def test_exact_match_wins_over_stripped_fallback():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="inspecteur Javert", type="personnage"))
    g.add_entity(Entity(name="Javert", type="personnage"))  # fusionné via repli

    assert len(g.entities) == 1


# --- 3. Rejet des enregistrements vides ---------------------------------------

def test_empty_entity_name_raises():
    with pytest.raises(ValueError):
        KnowledgeGraph().add_entity(Entity(name="   ", type="personnage"))


def test_incomplete_relation_raises():
    with pytest.raises(ValueError):
        KnowledgeGraph().add_relation(Relation(name="s'enfuit", source="Jean Valjean", target=""))


def test_sanitize_drops_llm_noise():
    # bruit réellement observé : entité au nom vide, relation sans cible
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="", type="inconnu"),
            ExtractedEntity(name="Jean Valjean", type="personnage"),
        ],
        relations=[
            ExtractedRelation(name="s'enfuit", source="Jean Valjean", target=""),
            ExtractedRelation(name="", source="Jean Valjean", target="Digne"),
            ExtractedRelation(name="arrive à", source="Jean Valjean", target="Digne"),
        ],
    )
    entities, relations = sanitize(result)
    assert [e.name for e in entities] == ["Jean Valjean"]
    assert [r.name for r in relations] == ["arrive à"]


def test_blank_entity_type_defaults_to_inconnu():
    g = KnowledgeGraph()
    e = g.add_entity(Entity(name="Digne", type="  "))
    assert e.type == "inconnu"
