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

## Recette : d'un roman à sa restitution graphique

Chaîne complète, de l'ingestion du texte à la page HTML interactive (exemple
Ollama local ; adapter `--model` / `--provider`) :

    # 1. Ingestion + extraction, pipeline qualité (canon_alias)
    minerva extract roman.txt -o roman.sqlite \
        --provider openai --base-url http://localhost:11434/v1 \
        --model gpt-oss:120b --refine

    # 2. Vérifier ce qui a été extrait
    minerva show roman.sqlite                    # entités + relations
    minerva show roman.sqlite --entity "Nom"     # détail + historique daté d'un attribut
    minerva timeline roman.sqlite                # ordre diégétique des moments

    # 3. (optionnel) Exporter le graphe brut en JSON
    minerva export roman.sqlite -o roman.json

    # 4. Restitution graphique : page HTML autonome (double-clic)
    minerva viz roman.sqlite -o roman.html

### À l'échelle d'un roman — ce qu'il faut savoir

Le pipeline est **validé sur des extraits courts** ; sur un roman entier, trois
points d'attention, non encore mesurés à cette échelle :

- **Temps.** L'extraction est séquentielle (un appel LLM par chunk). Sur
  gpt-oss:120b, compter **plusieurs heures** pour un roman ; `qwen3-coder-next`
  est ~3× plus rapide (qualité moindre, surtout sur le rappel des relations et
  les fusions) — arbitrage vitesse / qualité.
- **Cohérence inter-chunks.** Le prompt d'extraction ne rappelle qu'un plafond
  d'entités déjà vues (`MAX_KNOWN_ENTITIES_IN_PROMPT`, 200) ; au-delà, un
  personnage revu bien plus loin peut être ré-extrait en doublon — la
  canonicalisation (`--refine`) en rattrape une partie.
- **Visualisation.** La page `viz` précalcule chaque état du slider en Python ;
  à plusieurs centaines d'entités, le rendu et le slider peuvent devenir lourds.

Sans `--refine`, l'extraction nue reste l'option la plus sûre à très grande
échelle (elle découpe proprement en chunks, sans passe globale).

## Sous-commandes

### Extraction (`minerva extract`)

Analyse un fichier texte et écrit le graphe dans une base SQLite. Le texte est
découpé en chunks ; la cohérence inter-chunks (mêmes entités, timeline
continue) est gérée à la fusion.

    minerva extract roman.txt -o roman.sqlite \
        --provider openai --base-url http://localhost:11434/v1 --model gpt-oss:120b

Options : `--provider {anthropic,openai}` (défaut `anthropic`), `--model`,
`--base-url` (serveur compatible OpenAI, ex. Ollama), `--chunk-size`,
`--temperature` (backend `openai`/Ollama uniquement), et **`--refine`** — applique
le pipeline **canon_alias** (canonicalisation des coréférences puis passe
d'identité d'emprunt, à température 0), la meilleure qualité mesurée sur les
benchs. Sans `--refine`, l'extraction reste « nue » (rapide, mais ne fusionne pas
les alias ni les identités d'emprunt). La passe d'identité d'emprunt relit le
texte **par fenêtres** (elle ne charge donc pas tout le roman dans un seul prompt).
En sortie : le nombre d'entités, de relations et de moments écrits.

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
