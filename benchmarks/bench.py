"""Bench des modèles Ollama sur l'extraction minerva : temps + qualité vs référence.

Usage :
    .venv/bin/python benchmarks/bench.py [modèle ...]

Sans argument, teste la liste DEFAULT_MODELS. Les résultats (scores consolidés
+ graphe JSON par modèle) sont écrits dans benchmarks/results/<date>/.
Prérequis : un serveur Ollama local sur http://localhost:11434.
"""

import datetime
import json
import sys
import time
import traceback
from pathlib import Path

from minerva.extraction import extract_graph
from minerva.llm.openai_backend import OpenAIBackend
from minerva.model import KnowledgeGraph

HERE = Path(__file__).parent
TEXT = (HERE / "bench_texte.txt").read_text(encoding="utf-8")
CHUNK_SIZE = 700  # force plusieurs chunks pour tester la cohérence inter-chunks

DEFAULT_MODELS = [
    "qwen2.5:7b-instruct",
    "gemma4:latest",
    "qwen3.6:latest",
    "gpt-oss:20b",
    "mistral-small3.2:24b-instruct-2506-q8_0",
    "gpt-oss:120b",
]

# Entités attendues : nom -> variantes acceptables pour la résolution
EXPECTED_ENTITIES = {
    "Jean Valjean": ["Jean Valjean", "Valjean"],
    "Myriel": ["Myriel", "monseigneur Bienvenu", "l'évêque Myriel", "Mgr Bienvenu", "Bienvenu"],
    "Javert": ["Javert", "l'inspecteur Javert", "inspecteur Javert"],
    "Fantine": ["Fantine"],
    "Cosette": ["Cosette"],
    "Thénardier": ["Thénardier", "les Thénardier", "les époux Thénardier", "époux Thénardier"],
    "Toulon": ["Toulon", "bagne de Toulon"],
    "Digne": ["Digne"],
    "Montreuil-sur-Mer": ["Montreuil-sur-Mer"],
    "Montfermeil": ["Montfermeil"],
    "Paris": ["Paris"],
}

# Relations attendues : (variantes source, variantes cible) — le nom de la
# relation est libre, on vérifie seulement qu'un lien existe entre les deux.
EXPECTED_RELATIONS = [
    (EXPECTED_ENTITIES["Myriel"], EXPECTED_ENTITIES["Jean Valjean"]),
    (EXPECTED_ENTITIES["Fantine"], EXPECTED_ENTITIES["Cosette"]),
    (EXPECTED_ENTITIES["Thénardier"], EXPECTED_ENTITIES["Cosette"]),
    (EXPECTED_ENTITIES["Javert"], EXPECTED_ENTITIES["Jean Valjean"]),
    (EXPECTED_ENTITIES["Jean Valjean"], EXPECTED_ENTITIES["Cosette"]),
    (EXPECTED_ENTITIES["Jean Valjean"] + ["M. Madeleine", "Madeleine"], EXPECTED_ENTITIES["Fantine"]),
]

MADELEINE_VARIANTS = ["M. Madeleine", "Madeleine", "monsieur Madeleine"]


def resolve_any(graph: KnowledgeGraph, variants: list[str]):
    for v in variants:
        e = graph.resolve(v)
        if e is not None:
            return e
    return None


def score(graph: KnowledgeGraph) -> dict:
    entities_found = {
        name: resolve_any(graph, variants) is not None
        for name, variants in EXPECTED_ENTITIES.items()
    }

    def linked(src_variants, tgt_variants) -> bool:
        src = resolve_any(graph, src_variants)
        tgt = resolve_any(graph, tgt_variants)
        if src is None or tgt is None:
            return False
        return any(
            r.source in (src.name, tgt.name) and r.target in (src.name, tgt.name)
            for r in graph.relations
        )

    relations_found = [linked(s, t) for s, t in EXPECTED_RELATIONS]

    valjean = resolve_any(graph, EXPECTED_ENTITIES["Jean Valjean"])
    madeleine = resolve_any(graph, MADELEINE_VARIANTS)
    merged = valjean is not None and madeleine is not None and valjean is madeleine

    return {
        "n_entities": len(graph.entities),
        "n_relations": len(graph.relations),
        "n_entity_attrs": sum(len(e.attributes) for e in graph.entities),
        "n_relation_attrs": sum(len(r.attributes) for r in graph.relations),
        "expected_entities": sum(entities_found.values()),
        "expected_entities_total": len(entities_found),
        "missing_entities": [n for n, ok in entities_found.items() if not ok],
        "expected_relations": sum(relations_found),
        "expected_relations_total": len(relations_found),
        "valjean_madeleine_merged": merged,
    }


def main(models: list[str]) -> None:
    out_dir = HERE / "results" / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fusionne avec les résultats existants du jour : une nouvelle passe sur un
    # modèle remplace son entrée, les autres sont conservées.
    results_path = out_dir / "bench_results.json"
    previous: list[dict] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    )
    results: list[dict] = [e for e in previous if e.get("model") not in models]
    for model in models:
        print(f"=== {model} ===", flush=True)
        entry: dict = {"model": model}
        try:
            backend = OpenAIBackend(model=model, base_url="http://localhost:11434/v1")
            # échauffement : charge le modèle en mémoire hors chrono
            t0 = time.monotonic()
            backend._client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Réponds : ok"}]
            )
            entry["load_s"] = round(time.monotonic() - t0, 1)

            t0 = time.monotonic()
            graph = extract_graph(
                TEXT, backend, chunk_size=CHUNK_SIZE,
                on_progress=lambda d, t: print(f"  chunk {d}/{t}", flush=True),
            )
            entry["time_s"] = round(time.monotonic() - t0, 1)
            entry.update(score(graph))
            safe = model.replace(":", "_").replace("/", "_")
            (out_dir / f"bench_{safe}.json").write_text(
                json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TERMINÉ -> {out_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_MODELS)
