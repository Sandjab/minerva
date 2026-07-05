# Référence exhaustive précision/rappel sur textes inédits (axe 2)

Date : 2026-07-05. Validé par JP (corpus, périmètre, appariement, tolérance).

## Problème

Le scoring actuel (`benchmarks/bench.py`) mesure le **rappel seul** contre une
liste partielle (11 entités, 6 paires de relations), sur un extrait des
Misérables présent dans les données d'entraînement. Deux angles morts :

1. **Aucune précision** : un graphe qui hallucine 40 entités score autant
   qu'un graphe propre.
2. **Référence non exhaustive** : rien n'est dit du graphe au-delà des
   11 entités choisies.

L'axe 2 vise : textes hors données d'entraînement + référence exhaustive
permettant précision **et** rappel par couche.

## Décisions actées

| Décision | Choix |
|---|---|
| Corpus | Nouveau texte inédit riche (~2-3 pages) **et** `timeline_texte.txt`, tous deux annotés exhaustivement |
| Périmètre | Entités + alias (fusions) + relations. Attributs datés exclus (couverts par le volet timeline) |
| Appariement relations | Paire d'entités **non orientée**, prédicat ignoré. Déterministe, zéro juge LLM |
| Tolérance | Référence à deux niveaux : noyau (compte en rappel) + optionnel (ni rappel, ni faux positif) |

## Composants

### 1. Texte inédit `benchmarks/reference_texte.txt`

Écrit sur mesure pour ce bench (hors données d'entraînement), ~2-3 pages.
Contraintes de construction :

- 12 à 15 entités noyau (personnages, lieux, organisations) ;
- **une identité d'emprunt** (équivalent « M. Madeleine ») : un personnage
  désigné sous deux identités que le texte relie explicitement — le point
  faible connu du 120b ;
- surnoms et titres (« le docteur X », diminutifs) exigeant des fusions ;
- relations croisées entre au moins deux fils narratifs ;
- structure temporelle simple (pas de concurrence avec le volet timeline) ;
- des entités mineures défendables (pour peupler le niveau optionnel).

### 2. Références déclaratives `benchmarks/reference_<texte>.json`

Un fichier par texte : `reference_reference_texte.json` et
`reference_timeline_texte.json` (nom : `reference_` + nom du texte sans
extension). Schéma :

```json
{
  "text": "reference_texte.txt",
  "entities": [
    {"name": "Nom canonique", "type": "personne",
     "variants": ["Nom canonique", "surnom", "titre + nom"],
     "level": "core"}
  ],
  "required_merges": [
    ["mention identité A", "mention identité B"]
  ],
  "relations": [
    {"pair": ["Nom canonique 1", "Nom canonique 2"], "level": "core"}
  ]
}
```

- `entities[].level` : `core` ou `optional`. Les `variants` servent à
  l'appariement (normalisation partagée avec `minerva.model.normalize`).
- `relations[].pair` : noms canoniques d'entités **de la référence**
  (core ou optional), non orientés. `level` : `core` ou `optional`.
- `required_merges` : paires de mentions qui doivent résoudre vers la
  **même** entité prédite (généralise `valjean_madeleine_merged`).
- Exhaustivité : toute entité/relation défendable du texte figure dans le
  fichier, en `core` si un lecteur attentif la juge indispensable au graphe,
  en `optional` sinon.

### 3. Scorer `benchmarks/reference_scoring.py`

Module importable (utilisé par le harnais, `rescore.py` et les tests).
Entrée : `KnowledgeGraph` + référence chargée. Sortie : dict de métriques.

**Appariement entité prédite → entrée de référence** : une entité prédite
matche une entrée si l'une de ses mentions normalisées (`name` + `aliases`,
via `minerva.model.normalize`, y compris la variante article/titre dépouillée)
coïncide avec l'une des `variants` normalisées de l'entrée. Chaque entité
prédite matche au plus une entrée (la première dans l'ordre du fichier en cas
de collision — les variants doivent être disjoints entre entrées ; le scorer
émet un avertissement si une entité prédite matche plusieurs entrées).

**Entités** :
- VP : entrée `core` matchée par ≥ 1 entité prédite.
- Rappel = VP / nb entrées `core`.
- Doublon : 2ᵉ entité prédite (et suivantes) matchant une entrée déjà
  couverte → compte en **faux positif** (le graphe a un nœud de trop ; le
  défaut de fusion est aussi visible dans la métrique de fusions).
- Précision = (prédites matchant une entrée core ou optional, doublons
  exclus) / nb entités prédites.
- F1 = moyenne harmonique (0 si P + R = 0).

**Relations** : scoring sur l'ensemble des **paires non orientées
distinctes** du graphe prédit (les multi-prédicats entre deux mêmes entités
comptent pour une paire). Chaque extrémité est mappée vers son entrée de
référence (doublons inclus : un nœud dupliqué mappe vers la même entrée).
- Paire prédite = paire `core` → VP.
- Paire `optional`, ou impliquant une entité `optional` → neutre (compte
  comme correcte en précision, ignorée en rappel).
- Paire dont une extrémité ne mappe aucune entrée, ou paire absente de la
  référence → faux positif.
- Rappel = paires `core` couvertes / nb paires `core` ; précision =
  (VP + neutres) / nb paires prédites distinctes ; F1 idem.

**Fusions** : pour chaque paire de `required_merges`, réussie si les deux
mentions résolvent (`graph.resolve`) vers la même entité prédite non nulle.
Taux = réussies / total, plus le détail des paires ratées.

Métriques annexes conservées : `n_entities`, `n_relations`, listes
`missing_entities`, `false_positive_entities`, `missing_relations`,
`false_positive_relations` (pour l'inspection des graphes, workflow acté).

### 4. Harnais `benchmarks/bench_reference.py`

Mêmes conventions que `bench.py` : `--runs`, `--temperature`, warmup hors
chrono, fusion des résultats du jour par clé (modèle, température, runs,
pipeline, texte), graphes JSON sauvegardés dans `results/<date>/`,
agrégats moyenne/écart-type multi-runs. Spécificités :

- `--text` : `reference` (défaut) ou `timeline` ou `all` ;
- `--pipeline` : `nue` (extraction seule) ou `actee` (passe enrichie +
  complétude + canonicalisation, la reco du rapport) ou `all` ;
- CHUNK_SIZE 700 (cohérence inter-chunks, comme les autres benchs).

`rescore.py` étendu : sait rescorer un graphe sauvegardé contre une
référence exhaustive (les anciens graphes du bench NER restent rescorables
avec l'ancien scoring — pas de régression).

### 5. Campagne de bench

gpt-oss:120b et qwen3-coder-next, extraction nue et pipeline acté,
`--runs 3` minimum sur chaque texte. Livrable : addendum daté au
`rapport-2026-07-03.md` avec tableau P/R/F1 par couche × modèle × pipeline,
et lecture des faux positifs (inspection des graphes, pas des scores seuls).

## Tests

- **Scorer** (unitaires, sans LLM) : graphes synthétiques construits à la
  main dont P/R/F1 sont calculés à la main — cas : graphe parfait, entité
  hallucinée, doublon non fusionné, relation entre entités valides mais
  paire hors référence, relation vers entité optionnelle, relation vers
  entité inconnue, fusion ratée, graphe vide, collision de variants
  (avertissement).
- **Références** (validation) : test qui charge chaque JSON et vérifie la
  cohérence interne — variants disjoints entre entrées, paires de relations
  pointant vers des entrées existantes, `required_merges` résolvables dans
  les variants, niveaux valides.
- **Harnais** : pas de test LLM automatisé (bench manuel), mais la logique
  de fusion des résultats réutilise le motif éprouvé de `bench.py`.

## Critères de succès

1. Tests du scorer et des références verts (`pytest`).
2. Les deux textes annotés exhaustivement (relecture croisée : chaque
   entité nommée du texte est dans le JSON, core ou optional).
3. Campagne exécutée, addendum écrit avec tableau P/R/F1 et lecture.

## Hors périmètre

- Attributs datés dans la référence (volet timeline existant).
- Juge LLM pour les prédicats.
- Modification du pipeline d'extraction (si la campagne révèle un défaut,
  chantier séparé).
