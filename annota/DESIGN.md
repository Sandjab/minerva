# annota — atelier d'annotation d'entités & éval de fusion

**Date :** 2026-07-06 · **Statut :** conception validée, avant plan d'implémentation

## 1. Contexte et objectif

`minerva` extrait un graphe de connaissances d'un roman, puis `canon_alias`
(`--refine`) fusionne les surface forms coréférentes en entités canoniques
(`name` + `aliases`). On ne sait pas aujourd'hui **si cette fusion est correcte** :
elle peut **sur-fusionner** (réunir deux référents distincts) ou **sous-fusionner**
(laisser deux entités qui sont le même référent). Optimiser `canon_alias` sans
mesure, c'est travailler à l'aveugle.

`annota` produit la **vérité terrain** nécessaire à cette mesure — via une IHM
d'annotation qui montre le contexte pour désambiguïser — puis calcule l'écart
entre la partition de `canon_alias` et cette vérité (B³, LEA, décompte de paires).

### Constats sur les données réelles (base `out/roman.sqlite`, 370 entités)

Ces faits contraignent le design :

- **Le typage est inexploitable** : 354/370 entités sont `type='inconnu'`, seules
  8 sont `personnage` alors que de vrais personnages sont classés « inconnu ».
  → **pas de filtrage de périmètre par `type`**.
- **La sur-fusion est concentrée** : seules **23 entités ont ≥1 alias** (donc ont
  fusionné quelque chose). La sur-fusion ne peut venir que de là — bon marché à
  vérifier. Le sous-fusionnement, lui, est diffus sur les ~370 entités.
- **Aucun ancrage texte** : la base ne stocke ni le texte source ni de char-offset.
  Le seul lien vers le texte est `assertions.chunk_index`. → le contexte devra
  être reconstitué en **re-découpant le roman source**.

## 2. Périmètre

**Dans le périmètre (v1) :**
- Charger une base `.sqlite` minerva + le roman source.
- Présenter chaque **surface form** candidate (un `name` d'entité ou un `alias`)
  avec son contexte (passages du texte + attributs extraits).
- Actions d'annotation : **regrouper** des surface forms sous un `referent_id`
  unique ; leur attribuer un **type** ; **écarter** un candidat (bruit).
- Persister la vérité terrain de façon incrémentale (reprise possible).
- Calculer B³, LEA et le décompte de paires sur-/sous-fusionnées, en comparant la
  partition de `canon_alias` à la vérité terrain.

**Hors périmètre (YAGNI) :**
- Multi-utilisateur, authentification, base serveur, déploiement distant.
- Framework front avec étape de build (React, etc.).
- Plusieurs romans/bases en parallèle dans une même session.
- Mesure du **recall d'extraction** (entités jamais extraites) : `annota` évalue
  la **qualité du clustering**, pas la détection. Le bruit extrait est seulement
  compté comme indicateur, non pénalisé dans B³/LEA (surface forms `discard`).

## 3. Unité d'évaluation

L'unité atomique est la **surface form** : chaque `name` d'entité **et** chaque
`alias` de la base est un élément à part entière. On « dé-fusionne » ainsi les
décisions de `canon_alias` pour pouvoir les juger :

- **Partition prédite (P)** : deux surface forms sont dans le même cluster ssi
  elles appartiennent à la même entité dans la base (son `name` + ses `aliases`).
- **Partition vérité (G)** : le regroupement annoté par l'utilisateur.

Comparer P et G capture les deux erreurs : une surface form qu'un `alias` a
rattachée à tort révèle une **sur-fusion** ; deux entités distinctes que
l'utilisateur réunit sous un même `referent_id` révèlent une **sous-fusion**.

Les surface forms `discard` (bruit : dates, périphrases, fragments) sont retirées
de l'univers d'évaluation des deux côtés ; leur nombre est reporté à part.

## 4. Architecture — composants isolés

- **`reader`** — lit la base minerva (entités, alias, attributs, `chunk_index`)
  et re-découpe le source via `minerva.chunking` → liste de **candidats
  enrichis** (surface form, entity_id = cluster prédit, type prédit, attributs,
  passages de contexte). Lecture seule.
- **`store`** — la vérité terrain, persistée en SQLite (`gold.sqlite`) pour des
  écritures incrémentales atomiques. Table `annotations` : une décision par
  surface form.
- **`metrics`** — construit P (depuis la base) et G (depuis le store), calcule
  B³, LEA, le décompte de paires sur-/sous-fusionnées, et liste les paires
  fautives comme exemples concrets. MUC en option (illustre qu'il masque la
  sur-fusion). Pur, sans I/O réseau.
- **`server`** — serveur HTTP local (bibliothèque standard, zéro dépendance).
  Routes minimales : servir les candidats, poser/mettre à jour une décision,
  exporter le rapport de métriques.
- **`web/index.html`** — page unique HTML/CSS/JS vanilla inline : liste des
  candidats à gauche, **contexte à droite**, actions regrouper/typer/écarter,
  progression.

### Dépendance assumée

`annota` **importe `minerva.chunking`** (et lit le schéma SQLite de minerva).
Motif : garantir que le re-découpage retombe sur les mêmes chunks que
l'extraction, sinon `chunk_index` ne pointerait pas sur le bon passage. `minerva`
étant installé en `-e`, l'import est direct. `annota` reste un dossier séparé de
`src/minerva/`, extractible en repo dédié plus tard.

## 5. Modèle de données du gold (`gold.sqlite`)

```
annotations(
    surface_form TEXT NOT NULL,     -- le texte de la surface form
    entity_id    INTEGER NOT NULL,  -- entité minerva d'origine (cluster prédit)
    kind         TEXT NOT NULL,     -- 'name' | 'alias'
    referent_id  TEXT,              -- id de référent gold ; NULL si non décidé
    referent_type TEXT,             -- type gold optionnel (personnage/lieu/...)
    discarded    INTEGER DEFAULT 0, -- 1 = bruit, exclu de l'éval
    UNIQUE (surface_form, entity_id, kind)
)
```

`referent_id` est libre (une chaîne stable) : deux surface forms partageant le
même `referent_id` sont le même référent. Le `gold.sqlite` référence la base
minerva évaluée (nom/chemin en méta) pour éviter de croiser des annotations avec
la mauvaise base.

## 6. Flux

```
minerva .sqlite + roman source
        │
        ▼
     reader ──► candidats + contexte ──► server ──► web/index.html
                                                        │ (regrouper / typer / écarter)
                                                        ▼
                                                   store (gold.sqlite)
                                                        │
                                                        ▼
              metrics : P (base) vs G (gold) ──► B³ / LEA / paires + exemples
```

## 7. Stratégie de contexte

Pour une surface form, le contexte affiché combine, par ordre de fiabilité :

1. **Passages bruts du source** : depuis les `chunk_index` des assertions de son
   entité, re-découpe le roman (`minerva.chunking`, même `chunk_size` qu'à
   l'extraction) et affiche les chunks correspondants. C'est le signal le plus
   fidèle pour désambiguïser.
2. **Attributs extraits** de l'entité (`entity_attributes`) — déjà en base,
   souvent très discriminants (ex. une entité avec 12 attributs).
3. **Résumés de moments** liés (`appearances` → `moments`) — fallback court.

**Contrainte** : l'outil prend en argument le roman source et le `chunk_size`
(défaut = celui de minerva). Si les chunks reconstitués ne couvrent pas les
`chunk_index` observés (mauvais source ou mauvais `chunk_size`), l'outil
**avertit** et se rabat sur attributs + résumés, qui ne dépendent pas du
re-découpage.

## 8. Métriques

Toutes sur l'univers des surface forms annotées et non `discard`, à
l'intersection de P et G.

- **B³** : pour chaque surface form `m`, `précision = |P(m)∩G(m)| / |P(m)|`,
  `rappel = |P(m)∩G(m)| / |G(m)|` ; on moyenne, puis F1. Sensible à la
  sur-fusion (précision chute) comme à la sous-fusion (rappel chute).
- **LEA** (Moosavi & Strube 2016) : pondère chaque entité par sa taille et sa
  « résolvabilité » (liens internes) — plus discriminant que B³ sur les erreurs
  de clustering.
- **Décompte de paires** : nombre de paires de surface forms *co-groupées à tort*
  (sur-fusion) et *séparées à tort* (sous-fusion), avec la **liste des paires**
  comme exemples à inspecter — l'objet le plus actionnable pour corriger
  `canon_alias`.
- **MUC** (option) : reporté à côté pour montrer concrètement qu'il masque la
  sur-fusion (il est indulgent au sur-groupement).

Report annexe (hors B³/LEA) : nombre de surface forms `discard` = proxy du bruit
d'extraction.

## 9. Structure de fichiers

```
annota/
  DESIGN.md            # ce document
  README.md            # usage
  pyproject.toml       # projet séparé ; dépend de minerva (chunking) + stdlib
  annota/
    __init__.py
    reader.py
    store.py
    metrics.py
    server.py
    cli.py             # `annota serve <base.sqlite> --source roman.md` ; `annota score ...`
  web/
    index.html
  tests/
    test_metrics.py    # partitions jouets à valeurs attendues calculées à la main
    test_reader.py     # sur out/roman.sqlite : 421 surface forms, contexte d'une entité connue
```

## 10. Tests (intention, pas seulement comportement)

- **`metrics`** : cas jouets où le résultat est calculable à la main —
  partitions identiques (B³=1, 0 paire fautive) ; sur-fusion totale (rappel B³=1,
  précision effondrée, paires sur-fusionnées listées) ; sous-fusion totale
  (précision=1, rappel bas) ; discards bien exclus de l'univers. Chaque test
  encode *pourquoi* la métrique doit bouger (détecter tel type d'erreur), pas
  seulement une valeur.
- **`reader`** : sur `out/roman.sqlite`, produit bien 421 surface forms
  (370 names + 51 alias) ; pour une entité à attributs connus, renvoie ses
  attributs et au moins un passage de contexte ; avertit proprement sur mauvais
  `chunk_size`.

## 11. Risques et points résiduels

- **Coût d'annotation** : l'univers complet (421 surface forms) est lourd. On
  commence par les **23 clusters à alias** (sur-fusion, ~signal immédiat), le
  store et les métriques étant agnostiques au volume annoté (calcul sur ce qui
  est annoté). Extension aux « vraies » entités ensuite.
- **Reproductibilité du chunking** : dépend d'un `chunk_size` correct fourni par
  l'utilisateur (non stocké en base). Mitigé par l'avertissement + le fallback.
- **LEA** est plus délicate à implémenter que B³ ; si elle retarde la v1, B³ +
  décompte de paires suffisent au premier signal, LEA suit.
- **Motive une évolution minerva** : stocker les char-offsets des mentions
  (source grounding) rendrait le contexte exact et supprimerait le re-découpage.
  Hors périmètre `annota` v1, noté pour plus tard.
