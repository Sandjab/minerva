# Minerva — Modélisation des timelines (temps diégétique vs ordre de lecture)

Date : 2026-07-04 · Statut : approuvé · Branche : `timeline`

## Objectif

Minerva sert (entre autres) de base de départ à un futur outil de contrôle de
cohérence de romans français. Le contrôle de cohérence est hors périmètre, mais
la base livrée doit modéliser la **timeline** : distinguer le **temps
diégétique** (quand un fait est vrai dans l'histoire) de la **chronologie de
lecture** (ordre des chunks). Le récit linéaire fait coïncider les deux ;
flashbacks, ellipses (« vingt ans après ») et fils narratifs parallèles les
désynchronisent. Un attribut ou une relation peut changer entre deux moments
(Cosette : 8 ans → 18 ans) ; la distance temporelle entre deux constats doit
être exploitable en aval.

Cadre conceptuel : modélisation **bitemporelle** — le temps de lecture
(chunk) est l'analogue du *transaction time* (gratuit, totalement ordonné),
le temps diégétique celui du *valid time* (coûteux, partiellement ordonné).
Le temps diégétique d'un roman n'est pas un axe numérique mais un **graphe de
contraintes** (relations qualitatives type algèbre d'Allen + écarts quantifiés
quand le texte les donne).

## Décisions (validées avec l'utilisateur)

| Sujet | Décision |
|---|---|
| Périmètre | Capture des indices temporels **et** résolution en ordre partiel. Pas de détection de désynchros (outil aval). |
| Ancre diégétique | La **scène narrative** (« moment »), détectée par le LLM : 1..n moments par chunk. |
| Source de vérité | Journal de **constats** (assertions) datés bitemporellement ; les états classiques deviennent des vues dérivées. |
| Pipeline | **Passe unique enrichie** (le schéma de sortie LLM s'enrichit), benchée en non-régression contre l'existant. |
| Représentation du temps | Hybride : stockage en contraintes qualitatives + coordonnées **dérivées** par un résolveur en code (jamais le LLM). |

Contrainte de conception notée pour plus tard : une restitution graphique est
envisagée (forcegraph des entités + timeline façon banc de montage,
potentiellement une piste par personnage). Le modèle doit rendre ces vues
triviales à dériver ; la visualisation elle-même fera l'objet d'un design
dédié.

## Modèle de données (`minerva/model.py`)

Nouveaux concepts :

- `Moment` : scène narrative. `id` global, `chunk_index` + `seq` (position dans
  le chunk — l'ordre de lecture en découle), `summary` (une ligne).
- `TemporalConstraint` : arête du graphe temporel `(source, relation, target,
  gap)` avec `relation ∈ {avant, simultané, pendant}` et `gap` optionnel :
  `text` brut (« vingt ans après ») + `value`/`unit` fournis par le LLM,
  normalisés en jours **en code**.
- `Assertion` (constat) : fait daté bitemporellement. Sujet = entité **ou**
  relation, `attribute`, `value`, `moment_id` (temps diégétique),
  `chunk_index` (temps de lecture). L'observation de l'existence d'une
  relation à un moment est une assertion sans attribut.
- `Appearance` : présence d'une entité à un moment, même sans fait asserté
  (alimente la piste par personnage).

Reste **atemporel** : l'identité des entités (`name`, `type`, `aliases`) et des
relations (`name, source, target`). Toute la mécanique de résolution
(normalisation, alias, titres de civilité, index strippé) est conservée telle
quelle. Seuls les attributs et les observations deviennent temporels.

### Sémantique de fusion (déterministe, en code — jamais via LLM)

- « Première extraction gagne » disparaît du stockage : chaque fait devient une
  assertion. Déduplication stricte sur `(sujet, attribut, valeur, moment)`.
- Même attribut, valeurs différentes, moments différents → deux assertions
  (c'est le but).
- Même attribut, même moment, valeurs différentes → les deux sont conservées :
  minerva ne tranche pas, c'est la matière première de l'outil aval.
- Les refs de moments locales au chunk (`m1`, `m2`…) sont remappées vers des
  ids globaux à la fusion. Une ref de transition irrésoluble se replie sur
  « relatif au moment précédent en ordre de lecture ».

## Schéma de sortie LLM (passe unique enrichie)

`entities` au top niveau ne garde que l'identité ; les faits migrent dans des
blocs par moment :

```json
{
  "entities":  [{"name", "type", "aliases": [...]}],
  "moments": [{
    "ref": "m1",
    "summary": "Cosette, 8 ans, sert les Thénardier",
    "transition": {"type": "suite|flashback|retour|ellipse|parallèle",
                   "target": "id d'un moment connu (retour/parallèle), sinon null",
                   "gap": {"text": "vingt ans après", "value": 20, "unit": "années"}},
    "entities_present": ["Cosette", "Thénardier"],
    "facts": [
      {"entity": "Cosette", "attribute": "âge", "value": "8 ans"},
      {"relation": "héberge", "source": "Thénardier", "target": "Cosette"}
    ]
  }]
}
```

- Récit linéaire = coût quasi nul : un seul moment, `transition: suite`. Le cas
  nominal reste simple pour les petits modèles locaux.
- Chaque moment se positionne par défaut relativement au précédent en ordre de
  lecture ; les types de transition ne décrivent que les exceptions.
- Le prompt de chaque chunk inclut les derniers moments connus (id + summary,
  liste plafonnée) — extension directe du mécanisme existant pour les noms
  d'entités — afin de permettre `retour` et `parallèle` inter-chunks.
- Contrainte structured outputs inchangée (`additionalProperties: false`,
  paires plutôt que dicts libres).

Correspondance transition → contrainte (en code) :

| Transition | Contrainte générée (M = ce moment, P = précédent, T = target) |
|---|---|
| `suite` | P avant M |
| `ellipse` | P avant M, avec gap |
| `flashback` | M avant P |
| `retour` | T avant M (repli : P avant M) |
| `parallèle` | M simultané T (repli : aucune contrainte vs P) |

## Persistance SQLite (`minerva/store.py`)

```sql
-- inchangé : entities, entity_aliases, relations
moments(id PK, chunk_index, seq, summary,
        resolved_order INTEGER, resolved_days REAL)  -- dérivés, recalculables
moment_constraints(id PK, source_id, relation, target_id, gap_text, gap_days REAL)
appearances(moment_id, entity_id)
assertions(id PK, moment_id, entity_id, relation_id, attribute, value, chunk_index)
```

- `entity_attributes` et `relation_attributes` disparaissent au profit de vues
  SQL dérivées des assertions (état final, état première-valeur).
- `resolved_order` / `resolved_days` sont des colonnes dérivées, recalculables
  (documentées comme telles).
- Les `.db` legacy restent chargeables en mode dégradé (attributs → assertions
  à `moment_id` NULL) pour que `rescore.py` continue de fonctionner.

## Résolveur (`minerva/timeline.py`, code pur)

- Construit le DAG des `avant` ; les moments non contraints se replient sur
  l'ordre de lecture. Tri topologique **stable** (tie-break = ordre de
  lecture).
- Cycles (le LLM sera bruité) : signalés (fail loud dans les logs) et cassés
  au profit de l'ordre de lecture. Jamais de crash.
- Propagation des écarts quantifiés → `resolved_days` approximatif, avec pour
  origine (jour 0) le premier moment de l'ordre résolu ; renseigné seulement
  quand un chemin d'écarts quantifiés relie le moment à l'origine, `NULL`
  sinon. Jamais de précision inventée.
- Recalculable à volonté sur une base existante (équivalent timeline de
  `rescore.py`).

## API et CLI

- `KnowledgeGraph.state(policy="final"|"first", at=moment)` → snapshot
  classique `{attribut: valeur}` par entité/relation. `policy="first"` ≡
  comportement actuel (comparabilité bench et rescore).
- `KnowledgeGraph.track(entity)` → moments d'apparition + assertions,
  ordonnés par la résolution (la piste « banc de montage »).
- CLI : `minerva show --entity` affiche l'historique des valeurs ; `export`
  inclut moments, contraintes et assertions ; nouvelle sous-commande
  `minerva timeline graphe.db` (moments résolus, ordre et écarts).

## Tests (`tests/`, pytest — déterministes, sans réseau)

Remapping des moments à la fusion, construction des contraintes depuis les
transitions (tableau ci-dessus), déduplication d'assertions, résolveur
(cycles, écarts, branches parallèles, stabilité du tri), équivalence
`state("first")` ≡ ancien comportement de fusion, aller-retour SQLite
(y compris legacy en mode dégradé), orchestration d'extraction avec backend
factice enrichi.

## Bench (deux volets, addendum daté au rapport)

1. **Non-régression NER** : le schéma enrichi ne doit pas dégrader la qualité
   entités/relations vs l'existant, comparé via `state("first")` avec
   `bench.py --runs 5` sur les modèles de référence.
2. **Volet timeline** : texte fabriqué avec vérité terrain (flashbacks,
   ellipse « vingt ans après », deux fils parallèles) → précision/rappel sur
   les moments et les contraintes. Ce texte inédit sert aussi l'axe 2 du
   rapport (sortir du biais « Les Misérables dans les données
   d'entraînement »).

## Hors périmètre (volontairement)

Détection des désynchros temporelles (outil aval), restitution graphique
(design dédié ultérieur), datation calendaire absolue, raffinement du
résolveur au-delà du tri topologique + propagation simple des écarts.
