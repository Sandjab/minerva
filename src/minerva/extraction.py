"""Orchestration de l'extraction : chunking, appels LLM, fusion incrémentale.

Passe unique enrichie : chaque chunk renvoie ses moments narratifs (scènes),
les transitions qui les situent (suite/ellipse/flashback/retour/parallèle) et
les faits rattachés à chaque moment. Le code traduit les transitions en
contraintes temporelles et alimente le journal de constats — le LLM propose,
le code applique."""

from __future__ import annotations

import re
from collections.abc import Callable

from .chunking import DEFAULT_CHUNK_SIZE, split_text
from .llm import (
    ExtractedMoment,
    ExtractedTransition,
    ExtractionResult,
    LLMBackend,
    TimelineExtractionResult,
)
from .model import Assertion, Entity, KnowledgeGraph, Relation
from .timeline import AVANT, SIMULTANE, Gap, Moment, Timeline, gap_to_days

MAX_KNOWN_ENTITIES_IN_PROMPT = 200
MAX_KNOWN_MOMENTS_IN_PROMPT = 20
TRANSITION_TYPES = frozenset({"suite", "ellipse", "flashback", "retour", "parallèle"})

SYSTEM_PROMPT = """\
Tu es un extracteur d'informations narratives. On te donne un extrait de texte \
(par exemple d'un roman). Découpe l'extrait en « moments » narratifs (scènes), \
puis extrais les entités nommées, leurs attributs et les relations entre \
entités, chaque fait étant rattaché au moment où le texte l'affirme.

Règles sur les moments :
- Un moment = une scène dans un même cadre temporel. Un extrait linéaire \
simple = UN SEUL moment. Ne découpe que si le temps du récit change \
(flashback, saut en avant, retour au présent, fil parallèle).
- `ref` : identifiant local (« m1 », « m2 »...). `summary` : une ligne.
- `transition` situe le moment par rapport au moment PRÉCÉDENT (pour le \
premier moment de l'extrait : le dernier des « Moments déjà connus ») :
  - « suite » : continuation directe ou peu après (cas normal) ;
  - « ellipse » : plus tard, avec un saut notable (« vingt ans après ») ;
  - « flashback » : AVANT le moment précédent (souvenir, récit du passé) ;
  - « retour » : reprise d'un moment antérieur connu après un flashback — \
mets dans `target` l'identifiant du moment repris (ex. « M12 ») ;
  - « parallèle » : simultané à un autre fil — mets son identifiant dans \
`target` si tu le connais.
- `gap_text` : l'expression temporelle du texte (« le lendemain », « vingt \
ans après ») ; `gap_value` et `gap_unit` (heures/jours/semaines/mois/années) \
si le texte quantifie l'écart, sinon 0 et chaîne vide.
- `entities_present` : les entités présentes dans la scène.
- `facts` : chaque fait affirmé PAR CE MOMENT :
  - attribut d'entité : `entity` + `attribute` + `value` (ex. Cosette / âge / \
8 ans) ;
  - relation : `relation` + `source` + `target` (et `attribute`/`value` si la \
relation est qualifiée, ex. depuis / l'enfance).
  N'invente rien : uniquement ce que le texte dit.

Règles sur les entités :
- `name` : nom canonique le plus complet dans le texte (« Jean Valjean », pas \
« il »). Les autres désignations vont dans `aliases`.
- `type` : catégorie libre en français, en minuscules (ex. « personnage », \
« lieu », « objet », « organisation »).
- Noms d'attributs en français, en minuscules.
- Réutilise EXACTEMENT les noms des entités déjà connues et les identifiants \
des moments déjà connus quand l'extrait les concerne.\
"""

_MOMENT_REF = re.compile(r"^[mM](\d+)$")


def build_user_prompt(
    chunk: str, known_entities: list[str], known_moments: tuple[Moment, ...] = ()
) -> str:
    parts = []
    if known_entities:
        names = "\n".join(f"- {n}" for n in known_entities[:MAX_KNOWN_ENTITIES_IN_PROMPT])
        parts.append(f"Entités déjà connues :\n{names}")
    if known_moments:
        lines = "\n".join(f"- M{m.id} : {m.summary}" for m in known_moments)
        parts.append(f"Moments déjà connus (dans l'ordre du texte déjà lu) :\n{lines}")
    parts.append(f"Extrait à analyser :\n\n{chunk}")
    return "\n\n".join(parts)


def sanitize(result: ExtractionResult) -> tuple[list[Entity], list[Relation]]:
    """Écarte le bruit LLM : entités sans nom, relations incomplètes.
    (Utilisé par la passe de complétude de refine.py.)"""
    entities = [e.to_entity() for e in result.entities if e.name.strip()]
    relations = [
        r.to_relation()
        for r in result.relations
        if r.name.strip() and r.source.strip() and r.target.strip()
    ]
    return entities, relations


def _resolve_ref(ref: str, local: dict[str, int], timeline: Timeline) -> int | None:
    """Réf de transition -> id global : d'abord les refs locales au chunk,
    sinon « M<n> » global si le moment existe."""
    ref = ref.strip()
    if ref in local:
        return local[ref]
    match = _MOMENT_REF.match(ref)
    if match and timeline.moment(int(match.group(1))) is not None:
        return int(match.group(1))
    return None


def _apply_transition(
    timeline: Timeline,
    moment_id: int,
    transition: ExtractedTransition,
    prev_id: int | None,
    target_id: int | None,
) -> None:
    """Traduction transition -> contrainte (cf. table du spec). Type inconnu
    = « suite » ; sans moment précédent ni cible, aucune contrainte."""
    kind = transition.type.strip() if transition.type.strip() in TRANSITION_TYPES else "suite"
    gap = Gap(
        text=transition.gap_text.strip(),
        days=gap_to_days(transition.gap_value, transition.gap_unit),
    )
    if kind == "parallèle":
        if target_id is not None:
            timeline.add_constraint(moment_id, SIMULTANE, target_id)
        return  # sans cible : fil indépendant, aucune contrainte vs précédent
    if kind == "flashback":
        if prev_id is not None:
            timeline.add_constraint(moment_id, AVANT, prev_id, gap)
        return
    if kind == "retour":
        anchor = target_id if target_id is not None else prev_id
        if anchor is not None:
            timeline.add_constraint(anchor, AVANT, moment_id, gap)
        return
    # suite / ellipse
    if prev_id is not None:
        timeline.add_constraint(prev_id, AVANT, moment_id, gap)


def merge_chunk_result(
    graph: KnowledgeGraph, chunk_index: int, result: TimelineExtractionResult
) -> None:
    """Fusion déterministe d'un chunk : identités, moments (+ contraintes),
    présences et faits (-> journal). Écarte le bruit (champs vides)."""
    for identity in result.entities:
        if identity.name.strip():
            graph.add_entity(
                Entity(name=identity.name, type=identity.type, aliases=identity.aliases)
            )
    local: dict[str, int] = {}
    prev_id = graph.timeline.moments[-1].id if graph.timeline.moments else None
    for seq, extracted in enumerate(result.moments):
        moment = graph.timeline.add_moment(chunk_index, seq, extracted.summary)
        target_id = _resolve_ref(extracted.transition.target, local, graph.timeline)
        if target_id == moment.id:
            target_id = None
        _apply_transition(graph.timeline, moment.id, extracted.transition, prev_id, target_id)
        if extracted.ref.strip():
            local[extracted.ref.strip()] = moment.id
        _merge_moment_content(graph, chunk_index, moment.id, extracted)
        prev_id = moment.id


def _merge_moment_content(
    graph: KnowledgeGraph, chunk_index: int, moment_id: int, extracted: ExtractedMoment
) -> None:
    for name in extracted.entities_present:
        if not name.strip():
            continue
        entity = graph.resolve(name) or graph.add_entity(Entity(name=name, type="inconnu"))
        graph.timeline.add_appearance(moment_id, entity.name)
    for fact in extracted.facts:
        if fact.entity.strip():
            added = graph.add_assertion(
                Assertion(entity=fact.entity, attribute=fact.attribute, value=fact.value,
                          moment_id=moment_id, chunk_index=chunk_index)
            )
            if added is not None:
                graph.timeline.add_appearance(moment_id, added.entity)
        elif fact.relation.strip() and fact.source.strip() and fact.target.strip():
            added = graph.add_assertion(
                Assertion(relation_name=fact.relation, relation_source=fact.source,
                          relation_target=fact.target, attribute=fact.attribute,
                          value=fact.value, moment_id=moment_id, chunk_index=chunk_index)
            )
            if added is not None:
                graph.timeline.add_appearance(moment_id, added.relation_source)
                graph.timeline.add_appearance(moment_id, added.relation_target)


def extract_graph(
    text: str,
    backend: LLMBackend,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
) -> KnowledgeGraph:
    """Extrait le graphe complet d'un texte, chunk par chunk, timeline résolue."""
    graph = KnowledgeGraph()
    chunks = split_text(text, chunk_size)
    for i, chunk in enumerate(chunks):
        known = [e.name for e in graph.entities]
        recent = graph.timeline.recent(MAX_KNOWN_MOMENTS_IN_PROMPT)
        result = backend.parse(
            SYSTEM_PROMPT, build_user_prompt(chunk, known, recent), TimelineExtractionResult
        )
        merge_chunk_result(graph, i, result)
        if on_progress:
            on_progress(i + 1, len(chunks))
    graph.timeline.resolve()
    return graph
