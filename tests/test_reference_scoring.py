"""Tests du scoring exhaustif précision/rappel (benchmarks/reference_scoring.py)."""

import importlib.util
from pathlib import Path

from minerva.model import Entity, KnowledgeGraph, Relation

_BENCH = Path(__file__).parent.parent / "benchmarks"
_spec = importlib.util.spec_from_file_location(
    "reference_scoring", _BENCH / "reference_scoring.py"
)
scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring)

# Référence synthétique : 3 entités core, 1 optionnelle, 2 paires core,
# 1 paire optionnelle, 1 fusion exigée.
REF_DATA = {
    "text": "test.txt",
    "entities": [
        {"name": "Alice Vernet", "type": "personne",
         "variants": ["Alice Vernet", "Alice", "docteur Vernet"], "level": "core"},
        {"name": "Bruno Maillard", "type": "personne",
         "variants": ["Bruno Maillard", "Bruno"], "level": "core"},
        {"name": "Chaville", "type": "lieu", "variants": ["Chaville"], "level": "core"},
        {"name": "la Poste", "type": "organisation",
         "variants": ["la Poste", "Poste"], "level": "optional"},
    ],
    "required_merges": [[["Alice Vernet"], ["docteur Vernet"]]],
    "relations": [
        {"pair": ["Alice Vernet", "Bruno Maillard"], "level": "core"},
        {"pair": ["Alice Vernet", "Chaville"], "level": "core"},
        {"pair": ["Bruno Maillard", "Chaville"], "level": "optional"},
    ],
}


def make_ref():
    return scoring.Reference(REF_DATA)


def perfect_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne", aliases=["docteur Vernet"]))
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    g.add_entity(Entity(name="Chaville", type="lieu"))
    g.add_relation(Relation(name="connaît", source="Alice Vernet", target="Bruno Maillard"))
    g.add_relation(Relation(name="habite", source="Alice Vernet", target="Chaville"))
    return g


def test_graphe_parfait_entites():
    s = scoring.score_reference(perfect_graph(), make_ref())
    assert s["entity_precision"] == 1.0
    assert s["entity_recall"] == 1.0
    assert s["entity_f1"] == 1.0
    assert s["missing_entities"] == []
    assert s["false_positive_entities"] == []
    assert s["duplicate_entities"] == []


def test_entite_halucinee_compte_en_faux_positif():
    g = perfect_graph()
    g.add_entity(Entity(name="Zorglub", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["entity_precision"] == 0.75  # 3 matchées / 4 prédites
    assert s["entity_recall"] == 1.0
    assert s["false_positive_entities"] == ["Zorglub"]


def test_entite_optionnelle_neutre():
    # « la Poste » prédite : correcte en précision, absente du rappel.
    g = perfect_graph()
    g.add_entity(Entity(name="la Poste", type="organisation"))
    s = scoring.score_reference(g, make_ref())
    assert s["entity_precision"] == 1.0
    assert s["entity_recall"] == 1.0


def test_entite_manquante_baisse_le_rappel():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne"))
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["entity_recall"] == round(2 / 3, 3)
    assert s["missing_entities"] == ["Chaville"]
    assert s["entity_precision"] == 1.0


def test_doublon_non_fusionne_compte_en_faux_positif():
    # « docteur Vernet » créé comme entité séparée d'« Alice Vernet » (sans
    # alias posé, la résolution du graphe ne les relie pas) : même entrée de
    # référence couverte deux fois -> 1 nœud de trop.
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne"))
    g.add_entity(Entity(name="docteur Vernet", type="personne"))
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    g.add_entity(Entity(name="Chaville", type="lieu"))
    assert len(g.entities) == 4  # pas de fusion : « vernet » nu n'est indexé nulle part
    s = scoring.score_reference(g, make_ref())
    assert s["entity_precision"] == 0.75  # 3 entrées distinctes / 4 prédites
    assert s["entity_recall"] == 1.0
    assert s["duplicate_entities"] == ["Alice Vernet"]


def test_match_via_forme_sans_titre():
    # « commissaire Bruno Maillard » prédite : strip_title -> « Bruno Maillard ».
    g = KnowledgeGraph()
    g.add_entity(Entity(name="commissaire Bruno Maillard", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["false_positive_entities"] == []
    assert "Bruno Maillard" not in s["missing_entities"]


def test_graphe_vide():
    s = scoring.score_reference(KnowledgeGraph(), make_ref())
    assert s["entity_precision"] == 0.0
    assert s["entity_recall"] == 0.0
    assert s["entity_f1"] == 0.0
