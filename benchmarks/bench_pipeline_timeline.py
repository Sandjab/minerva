"""Bench du pipeline complet sur la passe enrichie (addendum 6, mitigation b).

Question : les passes de raffinement existantes (complétude, canonicalisation)
rattrapent-elles la régression NER de la passe d'extraction enrichie
(fusion d'alias 120b, relations coder-next) mesurée par bench.py ?

Usage :
    .venv/bin/python benchmarks/bench_pipeline_timeline.py [--runs N]

Pour chaque modèle (gpt-oss:120b, coder-next), N runs de :
extraction enrichie -> + complétude (temp 0) -> + canonicalisation (temp 0),
scorés à chaque étage avec le score() de bench.py (mêmes attentes que la
référence de l'addendum 5). Écrit results/<date>/pipeline_results.json.
"""

import datetime
import importlib.util
import json
import time
from pathlib import Path

from minerva.extraction import extract_graph
from minerva.llm.openai_backend import OpenAIBackend
from minerva.refine import canonicalize_graph, complete_graph

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
TEXT, CHUNK_SIZE, score = _bench.TEXT, _bench.CHUNK_SIZE, _bench.score

OLLAMA = "http://localhost:11434/v1"
MODELS = ["gpt-oss:120b", "qwen3-coder-next:latest"]


def backend(model: str, temperature: float | None = None) -> OpenAIBackend:
    return OpenAIBackend(model=model, base_url=OLLAMA, temperature=temperature)


def main(runs: int = 3) -> None:
    out_dir = HERE / "results" / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for model in MODELS:
        backend(model)._client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Réponds : ok"}]
        )
        safe = model.replace(":", "_").replace("/", "_")
        for i in range(1, runs + 1):
            t0 = time.monotonic()
            graph = extract_graph(
                TEXT, backend(model), CHUNK_SIZE,
                on_progress=lambda d, t: print(f"  {model} run {i} chunk {d}/{t}", flush=True),
            )
            t_extract = time.monotonic() - t0

            def record(stage: str, wall_s: float, detail: str = "") -> None:
                entry = {"model": model, "run": i, "stage": stage,
                         "wall_s": round(wall_s, 1), "detail": detail}
                entry.update(score(graph))
                results.append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)

            record("extraction", t_extract)

            t0 = time.monotonic()
            n = complete_graph(graph, TEXT, backend(model, temperature=0))
            t_complete = time.monotonic() - t0
            record("+complétude", t_extract + t_complete, detail=f"{n} items proposés")

            t0 = time.monotonic()
            before = len(graph.entities)
            graph = canonicalize_graph(graph, backend(model, temperature=0))
            t_canon = time.monotonic() - t0
            record("+canonicalisation", t_extract + t_complete + t_canon,
                   detail=f"entités {before}->{len(graph.entities)}")

            (out_dir / f"pipeline_{safe}_run{i}.json").write_text(
                json.dumps(graph.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    (out_dir / "pipeline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TERMINÉ", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline complet sur passe enrichie")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    main(runs=args.runs)
