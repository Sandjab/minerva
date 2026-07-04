"""L'ensemblisme doit garder ce sur quoi les modèles s'accordent (via alias et
paires orientées), écarter le bruit minoritaire, et ne solliciter l'arbitre
que sur le différend — c'est ce qui permet de combiner plusieurs modèles sans
hériter du bruit de chacun."""

from minerva.ensemble import ArbiterDecision, ArbiterResult, merge_arbitrated, merge_union, merge_vote
from minerva.model import Assertion, Entity, KnowledgeGraph, Relation


def graph_of(entities=(), relations=()):
    g = KnowledgeGraph()
    for e in entities:
        g.add_entity(e)
    for r in relations:
        g.add_relation(r)
    return g


def g_a():
    return graph_of(
        entities=[
            Entity(name="Jean Valjean", type="personnage", aliases=["M. Madeleine"],
                   attributes={"profession": "forçat"}),
            Entity(name="Cosette", type="personnage"),
            Entity(name="passeport jaune", type="objet"),  # bruit propre à A
        ],
        relations=[Relation(name="protège", source="Jean Valjean", target="Cosette")],
    )


def g_b():
    return graph_of(
        entities=[
            # même personne, désignée par l'alias : le vote doit la compter
            Entity(name="M. Madeleine", type="personnage", attributes={"statut": "maire"}),
            Entity(name="Cosette", type="personnage", attributes={"âge": "8 ans"}),
            Entity(name="Javert", type="personnage"),  # propre à B
        ],
        relations=[
            # même paire orientée que A, prédicat paraphrasé
            Relation(name="veille sur", source="M. Madeleine", target="Cosette"),
            Relation(name="poursuit", source="Javert", target="M. Madeleine"),
        ],
    )


def test_union_reduit_le_journal_aux_constats_non_dates():
    # Limitation documentée d'ensemble.py : les moments de deux modèles ne
    # s'alignent pas -> merge_union produit un graphe SANS timeline, les
    # constats datés retombant en constats non datés (moment_id None).
    a = KnowledgeGraph()
    ma = a.timeline.add_moment(chunk_index=0, seq=0, summary="chez les Thénardier")
    a.add_assertion(Assertion(entity="Cosette", attribute="âge", value="8 ans", moment_id=ma.id))

    b = KnowledgeGraph()
    mb = b.timeline.add_moment(chunk_index=0, seq=0, summary="dix ans après")
    b.add_assertion(Assertion(entity="Cosette", attribute="âge", value="18 ans", moment_id=mb.id))

    merged = merge_union([a, b])

    assert merged.timeline.moments == []
    assert merged.resolve("Cosette").attributes == {"âge": "8 ans"}  # vue first-wins intacte
    assert {(x.value, x.moment_id) for x in merged.assertions} == {("8 ans", None), ("18 ans", None)}


def test_union_merges_attributes_across_models():
    merged = merge_union([g_a(), g_b()])
    valjean = merged.resolve("Jean Valjean")
    assert valjean is not None
    assert valjean.attributes == {"profession": "forçat", "statut": "maire"}
    assert merged.resolve("passeport jaune") is not None  # l'union garde tout


def test_vote_keeps_consensus_and_drops_minority():
    merged = merge_vote([g_a(), g_b()], min_votes=2)
    names = {e.name for e in merged.entities}
    assert names == {"Jean Valjean", "Cosette"}  # accord via alias compris
    assert merged.resolve("passeport jaune") is None
    assert merged.resolve("Javert") is None


def test_vote_counts_relation_agreement_on_endpoints_not_predicate():
    merged = merge_vote([g_a(), g_b()], min_votes=2)
    pairs = {(r.source, r.target) for r in merged.relations}
    assert pairs == {("Jean Valjean", "Cosette")}
    # les deux prédicats paraphrasés sont conservés
    assert {r.name for r in merged.relations} == {"protège", "veille sur"}


def test_first_graph_wins_attribute_conflicts():
    a = graph_of([Entity(name="Cosette", type="personnage", attributes={"âge": "8 ans"})])
    b = graph_of([Entity(name="Cosette", type="personnage", attributes={"âge": "18 ans"})])
    merged = merge_vote([a, b], min_votes=2)
    assert merged.resolve("Cosette").attributes["âge"] == "8 ans"


class FakeArbiter:
    def __init__(self, keep_ids):
        self.keep_ids = set(keep_ids)
        self.prompts = []

    def parse(self, system, user, output_model):
        self.prompts.append(user)
        import re

        ids = re.findall(r"- ([ER]\d+) ", user)
        return ArbiterResult(
            decisions=[ArbiterDecision(id=i, keep=i in self.keep_ids) for i in ids]
        )


def test_arbitrated_submits_only_disputed_items():
    arbiter = FakeArbiter(keep_ids=[])
    merge_arbitrated([g_a(), g_b()], arbiter, text="texte source", min_votes=2)

    prompt = arbiter.prompts[0]
    assert "passeport jaune" in prompt and "Javert" in prompt  # litiges soumis
    # le consensus n'est pas soumis à l'arbitre
    assert "[entité] Jean Valjean" not in prompt
    assert "[entité] Cosette" not in prompt


def test_arbitrated_applies_keep_decisions():
    arbiter = FakeArbiter(keep_ids=["E1"])  # E0=passeport jaune, E1=Javert (ordre d'union)
    merged = merge_arbitrated([g_a(), g_b()], arbiter, text="t", min_votes=2)

    kept = {e.name for e in merged.entities}
    assert "Javert" in kept
    assert "passeport jaune" not in kept


def test_arbitrated_kept_relation_restores_its_endpoints():
    arbiter = FakeArbiter(keep_ids=["R0"])  # la seule paire en litige : Javert -> Valjean
    merged = merge_arbitrated([g_a(), g_b()], arbiter, text="t", min_votes=2)

    assert merged.resolve("Javert") is not None  # repêché car extrémité de R0
    assert any(r.name == "poursuit" for r in merged.relations)


def test_arbitrated_without_dispute_skips_the_llm():
    class ExplodingArbiter:
        def parse(self, system, user, output_model):
            raise AssertionError("l'arbitre ne doit pas être appelé sans litige")

    same = graph_of([Entity(name="Cosette", type="personnage")])
    merged = merge_arbitrated([same, graph_of([Entity(name="Cosette", type="personnage")])],
                              ExplodingArbiter(), text="t", min_votes=2)
    assert [e.name for e in merged.entities] == ["Cosette"]
