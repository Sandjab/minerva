"""L'orchestration fusionne les résultats de tous les chunks, transmet au
chunk suivant les entités ET les moments déjà connus, et traduit les
transitions narratives en contraintes temporelles — c'est le mécanisme qui
fait coïncider ordre de lecture et diégèse quand le récit est linéaire, et
les désynchronise sur flashback/ellipse/fil parallèle."""

from minerva.extraction import build_user_prompt, extract_graph
from minerva.llm import (
    ExtractedFact,
    ExtractedIdentity,
    ExtractedMoment,
    ExtractedTransition,
    TimelineExtractionResult,
)
from minerva.timeline import AVANT, SIMULTANE


class FakeBackend:
    def __init__(self, results):
        self._results = list(results)
        self.prompts = []

    def parse(self, system, user, output_model):
        self.prompts.append(user)
        return self._results.pop(0)


def _moment(ref, summary="", transition=None, present=(), facts=()):
    return ExtractedMoment(
        ref=ref, summary=summary,
        transition=transition or ExtractedTransition(),
        entities_present=list(present), facts=list(facts),
    )


def test_recit_lineaire_un_moment_par_chunk_contraintes_suite():
    r1 = TimelineExtractionResult(
        entities=[ExtractedIdentity(name="Cosette", type="personnage")],
        moments=[_moment("m1", "chez les Thénardier", present=["Cosette"],
                         facts=[ExtractedFact(entity="Cosette", attribute="âge", value="8 ans")])],
    )
    r2 = TimelineExtractionResult(
        moments=[_moment("m1", "dix ans après",
                         transition=ExtractedTransition(type="ellipse", gap_text="dix ans après",
                                                        gap_value=10, gap_unit="années"),
                         facts=[ExtractedFact(entity="Cosette", attribute="âge", value="18 ans")])],
    )
    backend = FakeBackend([r1, r2])
    graph = extract_graph("Un.\n\nDeux.", backend, chunk_size=5)

    assert len(graph.timeline.moments) == 2
    [c] = graph.timeline.constraints
    assert (c.source_id, c.relation, c.target_id) == (1, AVANT, 2)
    assert c.gap.days == 3650.0
    ages = [a.value for a in graph.assertions if a.attribute == "âge"]
    assert ages == ["8 ans", "18 ans"]
    assert graph.resolve("Cosette").attributes == {"âge": "8 ans"}  # vue first intacte
    assert graph.entity_state("final")["Cosette"]["âge"] == "18 ans"


def test_flashback_intra_chunk_genere_la_contrainte_inversee():
    r = TimelineExtractionResult(
        moments=[
            _moment("m1", "présent"),
            _moment("m2", "souvenir d'enfance",
                    transition=ExtractedTransition(type="flashback")),
        ],
    )
    graph = extract_graph("Texte.", FakeBackend([r]), chunk_size=100)
    [c] = graph.timeline.constraints
    assert (c.source_id, c.relation, c.target_id) == (2, AVANT, 1)  # M2 avant M1


def test_retour_reference_un_moment_connu_du_prompt():
    r1 = TimelineExtractionResult(moments=[_moment("m1", "fil principal")])
    r2 = TimelineExtractionResult(
        moments=[
            _moment("m1", "flashback", transition=ExtractedTransition(type="flashback")),
            _moment("m2", "retour au fil principal",
                    transition=ExtractedTransition(type="retour", target="M1")),
        ],
    )
    graph = extract_graph("Un.\n\nDeux.", FakeBackend([r1, r2]), chunk_size=5)
    # M1 (global) avant M3 : le retour pointe le moment repris, pas le flashback.
    assert any((c.source_id, c.relation, c.target_id) == (1, AVANT, 3)
               for c in graph.timeline.constraints)


def test_parallele_genere_simultane_et_ref_irresoluble_se_replie():
    r = TimelineExtractionResult(
        moments=[
            _moment("m1", "fil A"),
            _moment("m2", "fil B, au même moment",
                    transition=ExtractedTransition(type="parallèle", target="m1")),
            _moment("m3", "ref cassée",
                    transition=ExtractedTransition(type="retour", target="M99")),
        ],
    )
    graph = extract_graph("Texte.", FakeBackend([r]), chunk_size=100)
    kinds = {(c.source_id, c.relation, c.target_id) for c in graph.timeline.constraints}
    assert (2, SIMULTANE, 1) in kinds
    assert (2, AVANT, 3) in kinds  # repli : relatif au moment précédent


def test_parallele_sans_cible_ne_pose_aucune_contrainte():
    r = TimelineExtractionResult(
        moments=[
            _moment("m1", "fil A"),
            _moment("m2", "ailleurs, au même moment",
                    transition=ExtractedTransition(type="parallèle")),
        ],
    )
    graph = extract_graph("Texte.", FakeBackend([r]), chunk_size=100)
    assert graph.timeline.constraints == []


def test_faits_relationnels_crees_avec_observation_datee():
    r = TimelineExtractionResult(
        moments=[_moment("m1", "scène", facts=[
            ExtractedFact(relation="héberge", source="Thénardier", target="Cosette"),
            ExtractedFact(relation="héberge", source="Thénardier", target="Cosette",
                          attribute="contre", value="pension"),
        ])],
    )
    graph = extract_graph("Texte.", FakeBackend([r]), chunk_size=100)
    assert len(graph.relations) == 1
    rel = graph.relations[0]
    assert rel.attributes == {"contre": "pension"}
    assert len(graph.assertions) == 2  # observation nue + attribut


def test_presences_enregistrees_et_entites_creees_au_besoin():
    r = TimelineExtractionResult(
        moments=[_moment("m1", "scène", present=["Cosette", "Javert"])],
    )
    graph = extract_graph("Texte.", FakeBackend([r]), chunk_size=100)
    assert graph.timeline.appearances == {1: {"Cosette", "Javert"}}
    assert graph.resolve("Javert") is not None


def test_entites_et_moments_connus_transmis_au_chunk_suivant():
    r1 = TimelineExtractionResult(
        entities=[ExtractedIdentity(name="Cosette", type="personnage")],
        moments=[_moment("m1", "chez les Thénardier")],
    )
    r2 = TimelineExtractionResult()
    backend = FakeBackend([r1, r2])
    extract_graph("Un.\n\nDeux.", backend, chunk_size=5)
    assert "Cosette" not in backend.prompts[0]
    assert "Cosette" in backend.prompts[1]
    assert "M1" in backend.prompts[1]
    assert "chez les Thénardier" in backend.prompts[1]


def test_bruit_ecarte_faits_vides_et_transition_inconnue():
    r = TimelineExtractionResult(
        moments=[
            _moment("m1", "ok", transition=ExtractedTransition(type="n'importe quoi"),
                    facts=[ExtractedFact(entity="", attribute="âge", value="8"),
                           ExtractedFact(entity="X", attribute="", value="8"),
                           ExtractedFact(relation="aime", source="X", target="")]),
        ],
    )
    graph = extract_graph("Texte.", FakeBackend([r]), chunk_size=100)
    assert graph.assertions == []  # tout était incomplet
    assert graph.timeline.constraints == []  # type inconnu -> suite, sans précédent


def test_progress_callback_reporte_chaque_chunk():
    backend = FakeBackend([TimelineExtractionResult(), TimelineExtractionResult()])
    calls = []
    extract_graph("Un.\n\nDeux.", backend, chunk_size=5,
                  on_progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 2), (2, 2)]


def test_prompt_plafonne_entites_et_moments():
    known = [f"Entité {i}" for i in range(500)]
    from minerva.timeline import Timeline
    tl = Timeline()
    for i in range(50):
        tl.add_moment(i, 0, f"scène {i}")
    prompt = build_user_prompt("texte", known, tl.recent(20))
    assert "Entité 199" in prompt and "Entité 200" not in prompt
    assert "scène 49" in prompt and "scène 29" not in prompt


def test_chunk_sans_moment_rattache_ses_faits_hors_temps():
    # Un modèle faible peut ne renvoyer aucun moment : les entités restent.
    r = TimelineExtractionResult(
        entities=[ExtractedIdentity(name="Cosette", type="personnage")],
    )
    graph = extract_graph("Texte.", FakeBackend([r]), chunk_size=100)
    assert graph.resolve("Cosette") is not None
    assert graph.timeline.moments == []
