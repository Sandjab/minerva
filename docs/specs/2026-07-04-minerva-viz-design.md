# Minerva — Restitution graphique (forcegraph + timeline de montage)

Date : 2026-07-04 · Statut : approuvé · Branche : `viz`

## Objectif

Offrir une restitution visuelle d'une base minerva : un **forcegraph** des
entités/relations et une **timeline de montage** des moments (une piste par
entité, comme un banc de montage vidéo), couplées dans une même page. La viz
**consomme** la base et n'y touche pas : le modèle Python fait foi, le
JavaScript de la page ne fait que du rendu. Le contrôle de cohérence reste
hors périmètre de minerva.

Les choix de rendu ont été validés sur démos comparatives interactives
alimentées par les exports de bench réels (extrait des Misérables, 25 entités
/ 31 relations pour le graphe ; 13 moments / 11 pistes pour la timeline).

## Décisions (validées avec l'utilisateur)

| Sujet | Décision |
|---|---|
| Support | Page **HTML autonome** exportée par une sous-commande CLI (`minerva viz`). Zéro serveur, zéro CDN, s'ouvre d'un double-clic. |
| Lib forcegraph | **force-graph** (canvas 2D, ~250 Ko vendoré) — retenu sur démo contre d3-force SVG maison et 3d-force-graph WebGL. |
| Lib timeline | **d3 custom** (SVG, ~60–90 Ko vendoré) : layout **ordinal** fait main, fidèle au modèle — retenu sur démo contre vis-timeline, dont l'axe en dates ment (espacement uniforme pour des écarts inconnus). |
| Slider temporel | États du graphe **précalculés en Python** à l'export (`entity_state`/`relation_state(at=M)`), embarqués dans la page. Aucune logique métier dupliquée en JS. |
| Interactivité v1 | Les quatre : slider temporel, couplage des deux vues, tooltips riches, filtres. |
| Échelle | Chapitre en v1 (échelle des benchs), roman en cible : aucune décision qui bloque le passage au roman (canvas déjà choisi, scroll prévu, filtres présents), mais pas d'optimisation prématurée. |
| Validation | Tests automatisés (payload + export) **et** revue visuelle scénarisée par l'utilisateur sur les exports de bench. |
| `suite`/`retour` ≈ gap 0 | **Hors périmètre** : chantier modèle séparé (à bencher). La viz v1 est purement ordinale et en profitera automatiquement le jour venu. |

## Architecture

- **`minerva viz <base> -o page.html`** : nouvelle sous-commande dans
  `cli.py`, même patron que `export`.
- **`src/minerva/viz.py`** : construit le payload (voir ci-dessous), lit le
  template et les libs vendorées, injecte par remplacement de placeholders,
  écrit un seul fichier HTML. Stdlib pur (pas de Jinja), assets lus via
  `importlib.resources`.
- **`src/minerva/viz_assets/`** : `template.html` (structure, CSS, JS
  applicatif de rendu), `force-graph.min.js`, `d3.min.js` (vendorés dans le
  package, inclus au build hatch ; licences MIT/ISC conservées en tête des
  fichiers vendorés).
- Page : forcegraph en haut, timeline en dessous (scroll horizontal), slider
  commun. Poids attendu à l'échelle chapitre : ~350 Ko de libs + quelques
  dizaines de Ko de données.

## Payload embarqué (`window.MINERVA_DATA`)

Toutes les transformations se font en Python ; le JS reçoit des structures
prêtes à rendre.

- `entities` : nom, type, aliases, attributs → nœuds, couleur par type,
  tooltips.
- `relations` : source, cible, nom, attributs → liens + tooltips.
- `moments` : id, `resolved_order`, `resolved_days` (nullable), résumé →
  axe de la timeline, crans du slider.
- `gaps` : écarts quantifiés entre moments consécutifs (extraits des
  contraintes) → étiquettes `+N j` ; pointillé sinon.
- `tracks` : par entité, runs consécutifs de moments où elle apparaît
  (fusion des consécutifs précalculée) → clips de la timeline.
- `states` : par moment M, entités et relations visibles à M
  (`entity_state`/`relation_state(policy="final", at=M)`) → le slider ne
  fait que masquer/estomper. Signature exacte à vérifier dans le source à
  l'implémentation. Grossit en O(moments × entités) : acceptable en v1,
  passage en deltas seulement si l'échelle roman l'exige.
- `assertions` sources rattachées aux relations/attributs → tooltips riches.

## Interactions

- **Slider temporel** : cranté sur l'ordre résolu, résumé du moment courant
  affiché, défaut = dernier moment (graphe complet). Pilote les deux vues :
  nœuds/liens hors `states[M]` **estompés** (layout stable) dans le graphe,
  tête de lecture verticale sur la colonne M dans la timeline.
- **Couplage** : clic nœud → piste surlignée (scroll si besoin) ; clic piste
  → nœud + liens surlignés ; clic dans le vide → désélection.
- **Tooltips** : nœud → type, aliases, attributs ; lien → relation +
  assertions sources ; clip/colonne → résumé du moment + `j N` ou `j ?`.
- **Filtres** : cases par type d'entité, « masquer les apparitions uniques »,
  recherche par nom (aliases inclus). Appliqués aux deux vues en cohérence.

## Cas limites

- `resolved_days=None` partout sauf l'origine (cas nominal actuel : 12/13
  dans le run de démo) : layout ordinal, `j ?`, aucun crash.
- Fils parallèles non ordonnés entre eux : ordre résolu affiché tel quel
  (repli ordre de lecture), pas de signalement visuel en v1.
- Entité sans relation : nœud isolé, piste normale.
- Base sans moments : dégradation en forcegraph seul, bandeau « pas de
  timeline dans cette base ».

## Validation

**Tests automatisés** (pytest, `tests/`) — ils encodent « la page dit la
vérité du modèle » :

- Payload : toutes les entités/relations/moments/appearances de la base se
  retrouvent dans le payload ; runs de pistes corrects (fusion des
  consécutifs) ; `states` conformes à `entity_state`/`relation_state`.
- Cas limites : base sans moments, `resolved_days` tous `None`, entité
  isolée — export sans erreur, payload cohérent.
- Export : HTML autonome (aucune URL externe), payload et libs présents,
  tous les placeholders résolus.

**Revue visuelle scénarisée** (critère d'acceptation final, sur les exports
de bench) :

1. Je retrouve la piste de Jean Valjean et ses ruptures.
2. Le slider rejoue l'arrivée de Javert (absent avant son premier moment).
3. Le filtre « lieu » ne laisse que Digne / Montreuil-sur-Mer.
4. Clic sur Cosette → sa piste se surligne dans la timeline.

## Documentation

Section `minerva viz` dans le README (usage, options, capture ou GIF sur
l'extrait de bench), à la suite des sections existantes.

## Workflow

Branche dédiée `viz`, spec → plan → subagents (même workflow que le chantier
timelines). Merge sur main après validation des scénarios de revue visuelle.
