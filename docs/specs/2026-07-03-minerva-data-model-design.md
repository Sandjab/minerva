# Minerva — Modèle de données et extraction d'entités/relations

Date : 2026-07-03 · Statut : approuvé

## Objectif

Modéliser des **entités nommées** possédant des **attributs nommés dynamiques**, et des
**relations nommées** entre entités possédant elles aussi des attributs nommés.
Extraire d'un texte (ex. un roman) toutes les entités, relations et attributs
qu'il contient, via un LLM (API Claude ou API compatible OpenAI, ex. Ollama),
et persister le résultat dans SQLite. Interface : bibliothèque Python + CLI.

## Décisions (validées avec l'utilisateur)

| Sujet | Décision |
|---|---|
| Moteur d'extraction | LLM, double backend : SDK `anthropic` (Claude) + SDK `openai` avec `base_url` configurable (Ollama, etc.) |
| Schéma | Totalement dynamique : types, attributs et relations sont des chaînes libres découvertes dans le texte |
| Persistance | SQLite, schéma EAV (requêtable en SQL) |
| Interface | Package Python + CLI (`argparse`, sans dépendance lourde) |

## Modèle de données (`minerva/model.py`, Pydantic)

- `Entity` : `name` (canonique), `type` (libre), `aliases: list[str]`, `attributes: dict[str, str]`
- `Relation` : `name` (libre), `source` / `target` (noms canoniques d'entités), `attributes: dict[str, str]`
- `KnowledgeGraph` : entités indexées par nom normalisé + index d'alias, liste de relations.

### Sémantique de fusion (déterministe, en code — jamais via LLM)

- Résolution d'une entité : nom normalisé (minuscules, espaces réduits), puis index d'alias.
- Fusion d'entités : les attributs existants sont **préservés** (première extraction gagne),
  les nouveaux attributs et alias sont ajoutés. Le `type` existant est préservé.
- Fusion de relations : déduplication sur `(name, source, target)`, fusion des attributs
  avec la même règle.

## Persistance SQLite (`minerva/store.py`)

```sql
entities(id INTEGER PK, name TEXT UNIQUE, type TEXT)
entity_aliases(entity_id, alias)
entity_attributes(entity_id, name, value)
relations(id INTEGER PK, name TEXT, source_id, target_id)
relation_attributes(relation_id, name, value)
```

`save(graph, path)` écrase la base ; `load(path)` reconstruit un `KnowledgeGraph`.

## Extraction (`minerva/chunking.py`, `minerva/extraction.py`)

1. **Chunking** : découpage aux frontières de paragraphes, taille max configurable
   (défaut 8 000 caractères — compatible avec les contextes des modèles locaux).
2. **Appel LLM par chunk** avec sortie structurée (JSON Schema). Contrainte des
   structured outputs : `additionalProperties: false` obligatoire ⇒ les attributs
   sont transportés comme liste de paires `{name, value}` puis convertis en dict.
3. Le prompt de chaque chunk inclut la liste des noms d'entités déjà connues
   (plafonnée) pour favoriser la cohérence des noms canoniques entre chunks.
4. **Fusion incrémentale** dans le `KnowledgeGraph` après chaque chunk.

Schéma de sortie LLM :

```json
{
  "entities":  [{"name", "type", "aliases": [...], "attributes": [{"name", "value"}]}],
  "relations": [{"name", "source", "target", "attributes": [{"name", "value"}]}]
}
```

## Backends LLM (`minerva/llm/`)

- Protocole : `LLMBackend.extract(system: str, user: str) -> ExtractionResult`
- `AnthropicBackend` : `client.messages.parse(...)`, modèle par défaut `claude-opus-4-8`,
  adaptive thinking, `max_tokens=16000`.
- `OpenAIBackend` : `chat.completions.create(...)` avec `response_format` JSON Schema,
  `base_url` configurable (ex. `http://localhost:11434/v1` pour Ollama).
- Fabrique `make_backend(provider, model, base_url)`.

## CLI (`minerva/cli.py`)

```
minerva extract roman.txt -o graphe.db [--provider anthropic|openai] [--model M] [--base-url URL] [--chunk-size N]
minerva show graphe.db [--entity NOM]
minerva export graphe.db -o graphe.json
```

## Tests (`tests/`, pytest)

Déterministes, sans appel réseau : modèle et fusion, chunking, aller-retour SQLite,
orchestration d'extraction avec backend factice. Les backends réels ne sont pas
testés unitairement (fine couche sur les SDKs).

## Hors périmètre (volontairement)

Provenance fine (offsets dans le texte), résolution de coréférence avancée,
visualisation du graphe, parallélisation des appels LLM.
