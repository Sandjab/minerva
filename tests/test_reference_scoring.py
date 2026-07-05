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


def test_graphe_parfait_relations():
    s = scoring.score_reference(perfect_graph(), make_ref())
    assert s["relation_precision"] == 1.0
    assert s["relation_recall"] == 1.0
    assert s["missing_relations"] == []
    assert s["false_positive_relations"] == []


def test_relation_manquante_baisse_le_rappel():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne", aliases=["docteur Vernet"]))
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    g.add_entity(Entity(name="Chaville", type="lieu"))
    g.add_relation(Relation(name="connaît", source="Alice Vernet", target="Bruno Maillard"))
    s = scoring.score_reference(g, make_ref())
    assert s["relation_recall"] == 0.5
    assert s["missing_relations"] == [["Alice Vernet", "Chaville"]]
    assert s["relation_precision"] == 1.0


def test_paire_optionnelle_neutre():
    g = perfect_graph()
    g.add_relation(Relation(name="voisin", source="Bruno Maillard", target="Chaville"))
    s = scoring.score_reference(g, make_ref())
    assert s["relation_precision"] == 1.0  # (2 TP + 1 neutre) / 3
    assert s["relation_recall"] == 1.0


def test_relation_vers_entite_optionnelle_neutre():
    g = perfect_graph()
    g.add_entity(Entity(name="la Poste", type="organisation"))
    g.add_relation(Relation(name="travaille à", source="Bruno Maillard", target="la Poste"))
    s = scoring.score_reference(g, make_ref())
    assert s["relation_precision"] == 1.0
    assert s["relation_recall"] == 1.0


def test_paire_hors_reference_faux_positif():
    g = perfect_graph()
    g.add_relation(Relation(name="voisin", source="Bruno Maillard", target="Chaville"))
    g.add_entity(Entity(name="Zorglub", type="personne"))
    g.add_relation(Relation(name="menace", source="Zorglub", target="Alice Vernet"))
    s = scoring.score_reference(g, make_ref())
    # paires : {A,B} TP, {A,C} TP, {B,C} neutre, {Zorglub,A} FP -> 3/4
    assert s["relation_precision"] == 0.75
    assert s["false_positive_relations"] == [["Zorglub", "Alice Vernet"]]


def test_sens_et_predicat_ignores_et_multipredicats_dedupliques():
    g = perfect_graph()
    # sens inverse + deuxième prédicat sur la même paire : toujours 1 paire
    g.add_relation(Relation(name="emploie", source="Bruno Maillard", target="Alice Vernet"))
    s = scoring.score_reference(g, make_ref())
    assert s["relation_precision"] == 1.0
    assert s["relation_recall"] == 1.0


def test_fusion_reussie():
    s = scoring.score_reference(perfect_graph(), make_ref())
    assert s["merge_rate"] == "1/1"
    assert s["failed_merges"] == []


def test_fusion_ratee():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice Vernet", type="personne"))  # sans alias
    g.add_entity(Entity(name="Vernet", type="personne", aliases=["docteur Vernet"]))
    s = scoring.score_reference(g, make_ref())
    assert s["merge_rate"] == "0/1"
    assert s["failed_merges"] == [[["Alice Vernet"], ["docteur Vernet"]]]


def test_fusion_mention_absente_du_graphe():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Bruno Maillard", type="personne"))
    s = scoring.score_reference(g, make_ref())
    assert s["merge_rate"] == "0/1"


def test_validation_reference_saine():
    assert scoring.validate_reference(make_ref()) == []


def test_validation_detecte_les_incoherences():
    bad = {
        "text": "bad.txt",
        "entities": [
            {"name": "A", "type": "personne", "variants": ["A", "Double"], "level": "core"},
            {"name": "B", "type": "personne", "variants": ["B", "Double"], "level": "core"},
            {"name": "C", "type": "personne", "variants": [], "level": "exotique"},
        ],
        "required_merges": [[["Fantôme"], ["A"]], [["A"], ["B"]]],
        "relations": [
            {"pair": ["A", "Inconnu"], "level": "core"},
            {"pair": ["A", "A"], "level": "core"},
            {"pair": ["A", "B"], "level": "core"},
            {"pair": ["B", "A"], "level": "optional"},
        ],
    }
    errors = scoring.validate_reference(scoring.Reference(bad))
    text = "\n".join(errors)
    assert "Double" in text          # variant en collision
    assert "exotique" in text        # niveau invalide
    assert "aucun variant" in text   # C sans variant
    assert "Inconnu" in text         # paire vers entrée inexistante
    assert "['A', 'A']" in text      # paire réflexive
    assert "dupliquée" in text       # {A,B} deux fois
    assert "Fantôme" in text         # fusion hors variants
    assert "entrées différentes" in text  # fusion A/B inter-entrées
