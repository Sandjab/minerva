# annota — atelier d'annotation d'entités & éval de fusion

Outil local pour produire la **vérité terrain** de la fusion `canon_alias` de
minerva et **mesurer** son écart (B³, LEA, paires sur-/sous-fusionnées).

On annote des **surface forms** (le `name` d'une entité ou un de ses `alias`) en
les regroupant sous un identifiant de **référent**, avec le contexte du texte à
côté pour désambiguïser ; puis on compare ce regroupement à celui produit par
`canon_alias`. Voir [DESIGN.md](DESIGN.md) pour la conception.

## Prérequis

Le venv du projet minerva, avec `minerva` installé en editable (annota importe
`minerva.chunking`). Depuis `annota/` :

    pip install -e '.[dev]'

## Usage

### Annoter (atelier web)

    annota serve out/roman.sqlite --source in/roman.md --gold out/roman.gold.sqlite

Ouvre http://127.0.0.1:8000. Options : `--chunk-size` (défaut 8000, **doit valoir
celui utilisé à l'extraction** pour que les passages retombent sur le bon texte),
`--port`.

Dans l'atelier : la liste des candidats à gauche (groupés par entité prédite,
filtrée par défaut sur les **clusters à alias** — ceux à risque de sur-fusion),
le contexte à droite (attributs, passages sources surlignés, résumés). Pour
chaque candidat : assigner un **référent** (palette cliquable ou nouvel id),
un **type**, ou **écarter** le bruit. Raccourcis : `J`/`K` naviguer, `R`
référent, `1-6` type, `X` écarter, `Entrée` valider. Le bouton **Score** appelle
les métriques en direct. Chaque décision est persistée dans le gold.

### Mesurer canon_alias

    annota score out/roman.sqlite --gold out/roman.gold.sqlite

Affiche B³, LEA, le nombre de surface forms évaluées / écartées, et les paires
sur-/sous-fusionnées (exemples concrets à corriger).

## Périmètre

annota évalue la **qualité du clustering** (fusion), pas le recall d'extraction.
Le gold peut couvrir n'importe quel sous-ensemble : commence par les clusters à
alias (signal de sur-fusion immédiat), étends ensuite. Mono-utilisateur, local,
une base / un source / un gold à la fois.
