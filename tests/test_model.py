"""La fusion doit unifier les entités désignées différemment et ne jamais
perdre d'information déjà acquise — c'est ce qui rend l'extraction par chunks
cohérente sur un roman entier."""

from minerva.model import Entity, KnowledgeGraph, Relation


def test_add_entity_merges_by_normalized_name():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Jean Valjean", type="personnage", attributes={"âge": "25 ans"}))
    g.add_entity(Entity(name="  jean  valjean ", type="personnage", attributes={"profession": "forçat"}))

    assert len(g.entities) == 1
    e = g.resolve("jean valjean")
    assert e is not None
    assert e.attributes == {"âge": "25 ans", "profession": "forçat"}


def test_merge_preserves_existing_attribute_values():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Cosette", type="personnage", attributes={"âge": "8 ans"}))
    g.add_entity(Entity(name="Cosette", type="personnage", attributes={"âge": "18 ans"}))

    e = g.resolve("Cosette")
    assert e is not None and e.attributes["âge"] == "8 ans"  # première extraction gagne


def test_alias_resolves_to_canonical_entity():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Jean Valjean", type="personnage", aliases=["M. Madeleine"]))
    g.add_entity(Entity(name="M. Madeleine", type="personnage", attributes={"statut": "maire"}))

    assert len(g.entities) == 1
    e = g.resolve("Jean Valjean")
    assert e is not None and e.attributes == {"statut": "maire"}


def test_relation_endpoints_resolved_via_alias_and_created_if_missing():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Jean Valjean", type="personnage", aliases=["Valjean"]))
    g.add_relation(Relation(name="protège", source="Valjean", target="Cosette"))

    rel = g.relations[0]
    assert rel.source == "Jean Valjean"  # alias résolu en nom canonique
    assert g.resolve("Cosette") is not None  # extrémité manquante créée


def test_duplicate_relations_merge_attributes():
    g = KnowledgeGraph()
    g.add_relation(Relation(name="aime", source="Marius", target="Cosette", attributes={"depuis": "le jardin"}))
    g.add_relation(Relation(name="aime", source="Marius", target="Cosette", attributes={"intensité": "passionnée"}))

    assert len(g.relations) == 1
    assert g.relations[0].attributes == {"depuis": "le jardin", "intensité": "passionnée"}
