"""Bench précision/rappel contre les références exhaustives (axe 2).

Pour chaque (texte, pipeline, modèle) : extraction (temp. par défaut), puis
si pipeline « actee » : complétude + canonicalisation (temp 0, reco du
rapport). Scores P/R/F1 entités, relations, fusions — reference_scoring.py.

Usage :
    .venv/bin/python benchmarks/bench_reference.py [modèle ...] \
        [--text reference|timeline|all] [--pipeline nue|actee|all] [--runs N]
Prérequis : serveur Ollama local sur http://localhost:11434.
"""

import argparse
import datetime
import importlib.util
import json
import statistics
import time
import traceback
from pathlib import Path

from minerva.extraction import extract_graph
from minerva.llm.openai_backend import OpenAIBackend
from minerva.model import KnowledgeGraph
from minerva.refine import canonicalize_graph, complete_graph

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location(
    "reference_scoring", HERE / "reference_scoring.py"
)
_scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring)

OLLAMA = "http://localhost:11434/v1"
CHUNK_SIZE = 700  # plusieurs chunks : cohérence inter-chunks testée
DEFAULT_MODELS = ["gpt-oss:120b", "qwen3-coder-next:latest"]
TEXTS = {"reference": "reference_texte.txt", "timeline": "timeline_texte.txt"}
PIPELINES = ("nue", "actee")


def backend(model: str, temperature: float | None = None) -> OpenAIBackend:
    return OpenAIBackend(model=model, base_url=OLLAMA, temperature=temperature)


def run_one(model: str, text: str, pipeline: str, temperature: float | None,
            label: str) -> tuple[dict, KnowledgeGraph]:
    t0 = time.monotonic()
    graph = extract_graph(
        text, backend(model, temperature), chunk_size=CHUNK_SIZE,
        on_progress=lambda d, t: print(f"  {label} chunk {d}/{t}", flush=True),
    )
    detail = ""
    if pipeline == "actee":
        n = complete_graph(graph, text, backend(model, temperature=0))
        before = len(graph.entities)
        graph = canonicalize_graph(graph, backend(model, temperature=0))
        detail = f"complétude +{n}, canon {before}->{len(graph.entities)}"
    entry = {"time_s": round(time.monotonic() - t0, 1), "detail": detail}
    return entry, graph


AGG_KEYS = (
    "time_s", "entity_precision", "entity_recall", "entity_f1",
    "relation_precision", "relation_recall", "relation_f1",
)


def aggregate(per_run: list[dict]) -> dict:
    agg: dict = {"per_run": per_run}
    for key in AGG_KEYS:
        values = [r[key] for r in per_run]
        agg[f"{key}_mean"] = round(statistics.mean(values), 3)
        agg[f"{key}_std"] = round(statistics.stdev(values), 3) if len(values) > 1 else 0.0
    ok = sum(r["merges_ok"] for r in per_run)
    total = sum(r["merges_total"] for r in per_run)
    agg["merge_rate"] = f"{ok}/{total}"
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench P/R contre référence exhaustive")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--text", choices=[*TEXTS, "all"], default="reference")
    parser.add_argument("--pipeline", choices=[*PIPELINES, "all"], default="nue")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()
    models = args.models or DEFAULT_MODELS
    texts = list(TEXTS) if args.text == "all" else [args.text]
    pipelines = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]

    out_dir = HERE / "results" / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "reference_results.json"
    previous = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists() else []
    )

    def key(e: dict) -> tuple:
        return (e.get("model"), e.get("temperature"), e.get("runs"),
                e.get("text"), e.get("pipeline"))

    new_keys = {(m, args.temperature, args.runs, t, p)
                for m in models for t in texts for p in pipelines}
    results = [e for e in previous if key(e) not in new_keys]

    for model in models:
        # échauffement : charge le modèle hors chrono
        backend(model)._client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Réponds : ok"}]
        )
        safe = model.replace(":", "_").replace("/", "_")
        for text_key in texts:
            text = (HERE / TEXTS[text_key]).read_text(encoding="utf-8")
            ref = _scoring.load_reference(
                HERE / f"reference_{TEXTS[text_key].removesuffix('.txt')}.json"
            )
            for pipeline in pipelines:
                entry = {"model": model, "temperature": args.temperature,
                         "runs": args.runs, "text": text_key, "pipeline": pipeline}
                print(f"=== {model} / {text_key} / {pipeline} "
                      f"(runs={args.runs}) ===", flush=True)
                try:
                    per_run = []
                    for i in range(1, args.runs + 1):
                        label = f"{model} {text_key}/{pipeline} run {i}"
                        run_entry, graph = run_one(
                            model, text, pipeline, args.temperature, label
                        )
                        run_entry.update(_scoring.score_reference(graph, ref))
                        per_run.append(run_entry)
                        suffix = f"_run{i}" if args.runs > 1 else ""
                        (out_dir / f"ref_{text_key}_{pipeline}_{safe}{suffix}.json").write_text(
                            json.dumps(graph.to_dict(), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        print(json.dumps(
                            {k: run_entry[k] for k in (
                                "time_s", "entity_precision", "entity_recall",
                                "relation_precision", "relation_recall", "merge_rate")},
                            ensure_ascii=False), flush=True)
                    entry.update(per_run[0] if args.runs == 1 else aggregate(per_run))
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                results.append(entry)

    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TERMINÉ -> {results_path}", flush=True)


if __name__ == "__main__":
    main()
