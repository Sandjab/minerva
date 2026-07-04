# minerva
NER with dynamic attributes and rules

### Visualisation (`minerva viz`)

Exporte une page HTML **autonome** (zéro serveur, zéro réseau — s'ouvre d'un
double-clic) avec deux vues couplées :

- un **forcegraph** des entités et relations (couleur = type, taille = degré) ;
- une **timeline de montage** : une piste par entité, une colonne par moment
  en ordre diégétique résolu. Les écarts quantifiés s'affichent (`+3650 j`),
  les écarts inconnus restent des pointillés — l'axe ne ment jamais.

    minerva viz base.sqlite -o page.html

Interactions : slider temporel (rejoue l'état du graphe moment par moment,
précalculé en Python — le JS ne fait que du rendu), clic entité ↔ piste,
tooltips (aliases, attributs, assertions sources), filtres par type /
apparitions uniques / recherche.

La lib de rendu du graphe (force-graph, MIT) est vendorée dans le package ;
la timeline est en SVG vanilla.
