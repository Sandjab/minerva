# minerva
NER with dynamic attributes and rules

minerva lit un texte narratif et en extrait un **graphe de connaissances** :
entités nommées, relations, attributs datés et une **timeline diégétique**
(l'ordre de l'histoire, pas celui de la lecture). Le résultat est stocké dans
une base SQLite ; les sous-commandes ci-dessous la produisent puis l'explorent.

## Prérequis

- Python ≥ 3.10, puis `pip install -e .` (expose la commande `minerva`).
- Un backend LLM. Deux options :
  - **Ollama local** (chemin testé) : `--provider openai --base-url
    http://localhost:11434/v1 --model <modèle>` ;
  - **API Anthropic** : `--provider anthropic` (défaut), qui lit la clé dans
    l'environnement.

## Sous-commandes

### Extraction (`minerva extract`)

Analyse un fichier texte et écrit le graphe dans une base SQLite. Le texte est
découpé en chunks ; la cohérence inter-chunks (mêmes entités, timeline
continue) est gérée à la fusion.

    minerva extract roman.txt -o roman.sqlite \
        --provider openai --base-url http://localhost:11434/v1 --model gpt-oss:120b

Options : `--provider {anthropic,openai}` (défaut `anthropic`), `--model`,
`--base-url` (serveur compatible OpenAI, ex. Ollama), `--chunk-size`,
`--temperature` (backend `openai`/Ollama uniquement). En sortie : le nombre
d'entités, de relations et de moments écrits.

### Consultation (`minerva show`)

Affiche le contenu d'une base. Sans `--entity`, liste les entités (type,
nombre d'attributs) et les relations. Avec `--entity`, détaille une entité —
résolue par son nom **ou un de ses alias** — avec ses attributs, l'**historique
daté** d'un attribut qui change au fil du récit, et ses relations.

    minerva show roman.sqlite
    minerva show roman.sqlite --entity "Jean Valjean"

### Timeline (`minerva timeline`)

Affiche les moments narratifs résolus en ordre diégétique : ordre, jour résolu
quand il est quantifié, chunk de lecture d'origine et résumé, puis le nombre de
moments et de contraintes temporelles.

    minerva timeline roman.sqlite

### Export JSON (`minerva export`)

Exporte le graphe complet en JSON (entités, relations, moments, contraintes,
apparitions et journal d'assertions) — utile pour l'inspection ou un post-traitement.

    minerva export roman.sqlite -o roman.json

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
