"""Passes de raffinement d'un graphe extrait.

- Complétude : un LLM relit le texte face au graphe et ne renvoie que ce qui
  manque (entités, relations, attributs) ; la fusion existante fait le reste.
- Canonicalisation / coréférence : un LLM propose des groupes d'entités
  désignant le même référent (« évêque Myriel » = « Monseigneur Bienvenu ») ;
  l'application est déterministe et refuse les groupes incohérents.

Dans les deux cas le LLM propose, le code applique — jamais l'inverse.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field

from .chunking import DEFAULT_CHUNK_SIZE, split_text
from .extraction import sanitize
from .llm import ExtractionResult, LLMBackend
from .model import Entity, KnowledgeGraph

# Fenêtre de relecture de la passe alias : une passe qui met tout le texte dans
# un prompt ne tient pas à l'échelle roman (dépassement de contexte). On relit
# par fenêtres, la liste GLOBALE des entités étant fournie à chacune pour relier
# des mentions dispersées. Défaut = taille de chunk d'extraction.
ALIAS_WINDOW = DEFAULT_CHUNK_SIZE

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
        group_type = "inconnu"  # type de référence, comblé par le 1er type réel
        for name in group.members:
            entity = graph.resolve(name)
            if entity is None or entity.name in claimed or entity.name in members:
                continue
            # Garde-fou par type : on ne fusionne jamais deux types réels
            # différents (« personnage » ≠ « lieu »). « inconnu » est compatible
            # avec tout (il héritera du type). Le 1er type réel fixe la référence ;
            # un membre au type réel divergent est écarté, le reste du groupe fusionne.
            if entity.type != "inconnu":
                if group_type == "inconnu":
                    group_type = entity.type
                elif entity.type != group_type:
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
    window_size: int = ALIAS_WINDOW,
) -> KnowledgeGraph:
    """Relit le TEXTE pour proposer (LLM) puis appliquer (code) les fusions que
    la seule liste de noms ne révèle pas — au premier chef les identités
    d'emprunt. `scope` : « impersonation » (ciblé) ou « broad » (élargi aux
    périphrases narratives). Réutilise l'application déterministe des fusions.

    Passe à l'échelle : le texte est relu par fenêtres d'au plus `window_size`
    caractères (un appel LLM par fenêtre), la liste GLOBALE des entités étant
    fournie à chacune pour relier des mentions dispersées. Un texte plus court
    que la fenêtre = un seul appel (comportement d'origine). Les groupes de
    toutes les fenêtres sont cumulés puis appliqués une fois — `apply_canoni-
    calization` dédoublonne (garde-fou `claimed`)."""
    try:
        system = _ALIAS_SYSTEMS[scope]
    except KeyError:
        raise ValueError(
            f"scope inconnu : {scope!r} (attendu : {sorted(_ALIAS_SYSTEMS)})"
        )
    listing = _entity_listing(graph)  # globale, identique à chaque fenêtre
    groups: list[MergeGroup] = []
    for window in split_text(text, window_size):
        user = "Texte source :\n\n" + window + "\n\nEntités extraites :\n" + listing
        result = backend.parse(system, user, CanonicalizationResult)
        groups.extend(result.groups)
    return apply_canonicalization(graph, groups)


# --- Passe de typage des entités « inconnu » ----------------------------------
#
# La plupart des entités ne sont citées que dans les relations et les faits :
# elles sont auto-créées « inconnu » et jamais déclarées comme entités typées.
# Cette passe fait juger au LLM le type de chacune d'après son nom et son
# contexte (attributs, relations), puis l'applique via `add_entity` — qui comble
# « inconnu » sans écraser un vrai type. Le LLM propose, le code applique.

# Vocabulaire fermé des types d'entités. Une fois la couverture réglée (chaque
# entité reçoit un type), fermer la liste normalise les étiquettes : plus de
# « personnages » vs « personnage », et les sur-granularités (« boisson »,
# « meuble »...) se replient sur « objet ». Contrainte à deux niveaux : l'`enum`
# du json_schema (décodage contraint) ET le rappel dans le prompt. « autre » =
# soupape obligatoire, à n'employer que si aucun autre type ne convient.
EntityTypeName = Literal[
    "personnage", "lieu", "organisation", "objet", "document", "autre"
]
ENTITY_TYPES: tuple[str, ...] = get_args(EntityTypeName)

# Taille de lot de la passe de typage. À l'échelle roman, les entités « inconnu »
# se comptent par centaines ; les mettre toutes dans un seul prompt déborde le
# contexte et fait oublier des entités au modèle. On type par lots d'au plus
# TYPING_BATCH entités (un appel LLM par lot). Chaque entité étant typée
# indépendamment, d'après son propre nom+contexte, les lots sont DISJOINTS — pas
# de liste globale à répéter à chaque appel, contrairement à la passe alias
# (`ALIAS_WINDOW`) où une mention doit voir toutes les entités. Un graphe plus
# petit qu'un lot = un seul appel (comportement d'avant le fenêtrage).
TYPING_BATCH = 50

TYPING_SYSTEM = (
    "Tu assignes un TYPE à des entités déjà extraites d'un récit mais restées "
    "non typées. On te donne pour chacune son nom et ce qu'on sait d'elle "
    "(attributs, relations). Choisis le `type` de CHAQUE entité STRICTEMENT dans "
    "cette liste fermée : " + ", ".join(ENTITY_TYPES) + ". Déduis-le du nom et du "
    "contexte ; pour un humain nommé, « personnage » ; n'emploie « autre » que si "
    "aucun autre type ne convient. N'invente aucune entité."
)


class EntityType(BaseModel):
    name: str
    type: EntityTypeName


class TypingResult(BaseModel):
    types: list[EntityType] = Field(default_factory=list)


def _inconnu_lines(graph: KnowledgeGraph) -> list[str]:
    """Une ligne par entité « inconnu », avec son contexte typant (attributs,
    relations). Renvoyée en liste pour être découpée en lots (cf. `TYPING_BATCH`)."""
    lines = []
    for e in graph.entities:
        if e.type != "inconnu":
            continue
        attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items())
        rels = [
            f"{r.source} --{r.name}--> {r.target}"
            for r in graph.relations
            if e.name in (r.source, r.target)
        ]
        context = []
        if attrs:
            context.append("attributs : " + attrs)
        if rels:
            context.append("relations : " + " ; ".join(rels[:5]))
        suffix = f" ({' | '.join(context)})" if context else ""
        lines.append(f"- {e.name}{suffix}")
    return lines


def _type_pass(graph: KnowledgeGraph, backend: LLMBackend, batch_size: int) -> int:
    """Un passage de typage : les entités « inconnu » sont soumises au LLM par lots
    d'au plus `batch_size` (un appel par lot), les types de tous les lots étant
    appliqués ensemble. Renvoie le nombre d'entités effectivement typées.

    N'appelle pas le LLM s'il ne reste rien à typer. N'écrase jamais un vrai type
    et ignore un nom absent du graphe (garde-fous déterministes)."""
    lines = _inconnu_lines(graph)
    if not lines:
        return 0
    proposals: list[EntityType] = []
    for start in range(0, len(lines), batch_size):
        batch = "\n".join(lines[start : start + batch_size])
        result = backend.parse(TYPING_SYSTEM, "Entités à typer :\n" + batch, TypingResult)
        proposals.extend(result.types)
    applied = 0
    for proposed in proposals:
        incoming = proposed.type.strip().lower()
        if not incoming or incoming == "inconnu":
            continue
        entity = graph.resolve(proposed.name)
        if entity is None or entity.type != "inconnu":
            continue
        graph.add_entity(Entity(name=entity.name, type=incoming))  # comble « inconnu »
        applied += 1
    return applied


def type_entities(
    graph: KnowledgeGraph, backend: LLMBackend, *, batch_size: int = TYPING_BATCH
) -> int:
    """Type en place les entités restées « inconnu ». Renvoie le nombre total typé.

    Boucle jusqu'à convergence : re-typer les « inconnu » restants tant qu'un
    passage en type au moins un. Face à des centaines d'entités, le modèle en
    oublie dans un lot ; re-soumis, l'oublié tombe dans un lot différent (plus
    petit) et se fait typer — le résiduel d'un run à l'échelle roman s'efface ainsi.
    Terminaison garantie : le nombre d'« inconnu » décroît strictement à chaque
    passage productif ; un passage sans progrès (0 inconnu, ou restants
    irrécupérables) arrête la boucle."""
    total = 0
    while (applied := _type_pass(graph, backend, batch_size)) > 0:
        total += applied
    return total


# --- Pipeline de raffinement ---------------------------------------------------

def refine_graph(graph: KnowledgeGraph, text: str, backend: LLMBackend) -> KnowledgeGraph:
    """Enchaîne les passes de raffinement dans l'ordre : canonicalisation
    (coréférence sur les noms), alias (identités d'emprunt révélées par le
    texte), puis typage des entités restées « inconnu ». Le typage vient EN
    DERNIER : il opère sur le jeu d'entités final (après fusions), donc sans
    typer une entité qui aurait ensuite fusionné. Renvoie le graphe raffiné."""
    graph = canonicalize_graph(graph, backend)
    graph = resolve_aliases(graph, text, backend)
    type_entities(graph, backend)
    return graph
