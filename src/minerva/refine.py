"""Passes de raffinement d'un graphe extrait.

- Complétude : un LLM relit le texte face au graphe et ne renvoie que ce qui
  manque (entités, relations, attributs) ; la fusion existante fait le reste.
- Canonicalisation / coréférence : un LLM propose des groupes d'entités
  désignant le même référent (« évêque Myriel » = « Monseigneur Bienvenu ») ;
  l'application est déterministe et refuse les groupes incohérents.

Dans les deux cas le LLM propose, le code applique — jamais l'inverse.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .extraction import sanitize
from .llm import ExtractionResult, LLMBackend
from .model import KnowledgeGraph

# --- Passe de complétude -------------------------------------------------------

COMPLETION_SYSTEM = """\
Tu es un vérificateur de complétude d'extraction d'information. On te donne un \
texte source et le graphe d'entités/relations déjà extrait. Relis le texte et \
renvoie UNIQUEMENT ce qui manque au graphe :
- les entités nommées absentes ;
- les relations affirmées par le texte entre entités (existantes ou nouvelles) \
absentes du graphe ;
- les attributs manquants d'entités existantes (renvoie alors l'entité avec son \
nom EXACT tel qu'il figure dans le graphe, et seulement les attributs nouveaux).
Ne renvoie RIEN de ce qui est déjà présent. Si rien ne manque, renvoie des \
listes vides. Mêmes conventions que l'extraction : noms canoniques complets, \
types et attributs en français minuscules, rien d'inventé.\
"""


def _graph_summary(graph: KnowledgeGraph) -> str:
    lines = ["Entités :"]
    for e in graph.entities:
        attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items())
        alias = f" (alias : {', '.join(e.aliases)})" if e.aliases else ""
        lines.append(f"- {e.name} [{e.type}]{alias}{' — ' + attrs if attrs else ''}")
    lines.append("Relations :")
    for r in graph.relations:
        lines.append(f"- {r.source} --{r.name}--> {r.target}")
    return "\n".join(lines)


def complete_graph(graph: KnowledgeGraph, text: str, backend: LLMBackend) -> int:
    """Complète `graph` en place ; renvoie le nombre d'items proposés (bruts)."""
    user = (
        "Texte source :\n\n" + text
        + "\n\nGraphe déjà extrait :\n" + _graph_summary(graph)
    )
    result = backend.parse(COMPLETION_SYSTEM, user, ExtractionResult)
    entities, relations = sanitize(result)
    graph.merge(entities, relations)
    return len(entities) + len(relations)


# --- Passe de canonicalisation / coréférence ----------------------------------

CANONICALIZATION_SYSTEM = """\
Tu es un résolveur de coréférence au niveau d'un graphe d'entités. On te donne \
la liste des entités extraites d'un texte. Identifie les groupes d'entités qui \
désignent LE MÊME référent sous des noms différents (titre vs nom, périphrase, \
variante orthographique, ex. « évêque Myriel » et « monseigneur Bienvenu »).
Pour chaque groupe, choisis comme `canonical` le nom le plus complet et le plus \
précis PARMI les membres, et liste dans `members` TOUS les noms du groupe \
(y compris le canonique). Ne groupe JAMAIS deux personnes différentes (ex. \
« M. Thénardier » et « Mme Thénardier » restent distincts). Ne renvoie que les \
groupes d'au moins deux membres ; s'il n'y a rien à fusionner, renvoie une \
liste vide.\
"""


class MergeGroup(BaseModel):
    canonical: str
    members: list[str] = Field(default_factory=list)


class CanonicalizationResult(BaseModel):
    groups: list[MergeGroup] = Field(default_factory=list)


def _entity_listing(graph: KnowledgeGraph) -> str:
    lines = []
    for e in graph.entities:
        alias = f" (alias : {', '.join(e.aliases)})" if e.aliases else ""
        lines.append(f"- {e.name} [{e.type}]{alias}")
    return "\n".join(lines)


def apply_canonicalization(
    graph: KnowledgeGraph, groups: list[MergeGroup]
) -> KnowledgeGraph:
    """Reconstruit le graphe en fusionnant les groupes proposés.

    Garde-fous déterministes : un membre inconnu du graphe est ignoré ; un
    membre déjà pris par un groupe précédent est ignoré ; un groupe réduit à
    moins de deux membres effectifs ne fusionne rien ; le canonique doit être
    l'un des membres (sinon le premier membre valide fait canonique).
    """
    claimed: set[str] = set()
    plan: list[tuple[str, list[str]]] = []  # (nom canonique, autres membres)
    for group in groups:
        members: list[str] = []
        for name in group.members:
            entity = graph.resolve(name)
            if entity is None or entity.name in claimed or entity.name in members:
                continue
            members.append(entity.name)
        if len(members) < 2:
            continue
        canonical_entity = graph.resolve(group.canonical)
        canonical = (
            canonical_entity.name
            if canonical_entity is not None and canonical_entity.name in members
            else members[0]
        )
        others = [m for m in members if m != canonical]
        claimed.update(members)
        plan.append((canonical, others))

    merged = KnowledgeGraph()
    # 1. Les canoniques d'abord, avec les autres membres déclarés comme alias :
    #    tout ce qui arrive ensuite sous un nom de membre fusionne dedans.
    #    Identités seules : les attributs sont rejoués via le journal (étape 3),
    #    sinon ils seraient dégradés en constats non datés.
    for canonical, others in plan:
        entity = graph.resolve(canonical)
        assert entity is not None
        copy = entity.model_copy(deep=True)
        copy.aliases = list(dict.fromkeys(copy.aliases + others))
        copy.attributes = {}
        merged.add_entity(copy)
    # 2. Le reste du graphe passe par la fusion normale (alias résolus).
    for entity in graph.entities:
        copy = entity.model_copy(deep=True)
        copy.attributes = {}
        merged.add_entity(copy)
    for relation in graph.relations:
        copy = relation.model_copy(deep=True)
        copy.attributes = {}
        merged.add_relation(copy)
    # 3. Le journal est rejoué avec résolution des nouveaux canoniques :
    #    dicts d'attributs reconstruits, moments et présences conservés.
    merged.adopt_journal(graph)
    return merged


def canonicalize_graph(graph: KnowledgeGraph, backend: LLMBackend) -> KnowledgeGraph:
    """Propose (LLM) puis applique (code) la fusion des entités coréférentes."""
    result = backend.parse(
        CANONICALIZATION_SYSTEM,
        "Entités extraites :\n" + _entity_listing(graph),
        CanonicalizationResult,
    )
    return apply_canonicalization(graph, result.groups)


# --- Passe d'alias / identité d'emprunt (relit le TEXTE) ----------------------
#
# La canonicalisation ne voit que les NOMS ; elle ne peut donc pas relier deux
# noms propres distincts (« Antoine Sérac » et « Théo Rivière ») que seul le
# récit désigne comme une même personne. Cette passe fournit le texte au modèle
# et lui demande les fusions que la lecture — et elle seule — révèle. Le format
# de sortie et l'application déterministe sont ceux de la canonicalisation.

IMPERSONATION_SYSTEM = """\
Tu repères les IDENTITÉS D'EMPRUNT dans un récit. On te donne le texte source et \
la liste des entités déjà extraites. Cherche dans le TEXTE les cas où un même \
individu porte deux noms propres différents parce qu'il cache ou change \
d'identité : pseudonyme, fausse identité, nom d'emprunt, alias révélé (« il \
s'appelait en réalité X », « X, alias Y », double signature « X — Y »). Ne te \
fie PAS à la ressemblance des noms : ici les deux noms sont volontairement \
distincts, seule la lecture du texte permet de les relier. Pour chaque identité, \
renvoie un groupe : `canonical` = le nom sous lequel le personnage est le plus \
souvent désigné dans le récit, `members` = TOUS ses noms (le canonique inclus). \
Ne relie JAMAIS deux personnes réellement distinctes. Si le texte ne révèle \
aucune identité d'emprunt, renvoie une liste vide.\
"""

BROAD_ALIAS_SYSTEM = """\
Tu es un résolveur de coréférence qui s'appuie sur le TEXTE. On te donne le texte \
source et la liste des entités déjà extraites. En relisant le texte, identifie \
tous les groupes d'entités qui désignent le MÊME référent mais que la seule liste \
de noms ne permet pas de rapprocher :
- identités d'emprunt (pseudonyme, fausse identité, « il s'appelait en réalité \
X », « X, alias Y », double signature) ;
- périphrases et désignations narratives qu'un nom propre du texte identifie sans \
ambiguïté (« le cartographe », « le géomètre » renvoyant à la personne nommée).
Pour chaque groupe, `canonical` = le nom le plus complet et le plus fréquent, \
`members` = TOUS les noms du groupe (le canonique inclus). Ne relie JAMAIS deux \
référents réellement distincts ni une périphrase ambiguë. Si rien n'est à relier, \
renvoie une liste vide.\
"""

_ALIAS_SYSTEMS = {
    "impersonation": IMPERSONATION_SYSTEM,
    "broad": BROAD_ALIAS_SYSTEM,
}


def resolve_aliases(
    graph: KnowledgeGraph,
    text: str,
    backend: LLMBackend,
    *,
    scope: str = "impersonation",
) -> KnowledgeGraph:
    """Relit le TEXTE pour proposer (LLM) puis appliquer (code) les fusions que
    la seule liste de noms ne révèle pas — au premier chef les identités
    d'emprunt. `scope` : « impersonation » (ciblé) ou « broad » (élargi aux
    périphrases narratives). Réutilise l'application déterministe des fusions."""
    try:
        system = _ALIAS_SYSTEMS[scope]
    except KeyError:
        raise ValueError(
            f"scope inconnu : {scope!r} (attendu : {sorted(_ALIAS_SYSTEMS)})"
        )
    user = (
        "Texte source :\n\n" + text
        + "\n\nEntités extraites :\n" + _entity_listing(graph)
    )
    result = backend.parse(system, user, CanonicalizationResult)
    return apply_canonicalization(graph, result.groups)
