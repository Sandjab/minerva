"""Bench des passes de raffinement : complétude puis canonicalisation/coref.

Usage :
    .venv/bin/python benchmarks/bench_refine.py

Configs :
- efficacité : coder-next seul -> + complétude -> + canonicalisation ;
- bruit : union des 3 petits -> + canonicalisation ;
- exactitude : union(120b, coder-next) -> + canonicalisation.
Écrit benchmarks/results/<date>/refine_results.json.
"""

import concurrent.futures
import datetime
import importlib.util
import json
import time
from pathlib import Path

from minerva.ensemble import merge_union
from minerva.extraction import extract_graph
from minerva.llm.openai_backend import OpenAIBackend
from minerva.refine import canonicalize_graph, complete_graph

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
TEXT, CHUNK_SIZE, score = _bench.TEXT, _bench.CHUNK_SIZE, _bench.score

OLLAMA = "http://localhost:11434/v1"
MID = "qwen3-coder-next:latest"
BIG = "gpt-oss:120b"
SMALLS = ["gpt-oss:20b", "gemma4:latest", "qwen2.5:7b-instruct"]


def backend(model: str, temperature: float | None = None) -> OpenAIBackend:
    return OpenAIBackend(model=model, base_url=OLLAMA, temperature=temperature)


def warmup(models: list[str]) -> None:
    for m in models:
        backend(m)._client.chat.completions.create(
            model=m, messages=[{"role": "user", "content": "Réponds : ok"}]
        )


def main() -> None:
    out_dir = HERE / "results" / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def record(name, graph, wall_s, detail=""):
        entry = {"config": name, "wall_s": round(wall_s, 1), "detail": detail}
        entry.update(score(graph))
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        safe = name.replace(" ", "_").replace("/", "_")
        (out_dir / f"refine_{safe}.json").write_text(
            json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Voie efficacité : coder-next + complétude + canonicalisation ---------
    warmup([MID])
    t0 = time.monotonic()
    graph = extract_graph(TEXT, backend(MID), CHUNK_SIZE)
    t_extract = time.monotonic() - t0
    record("coder-next/seul", graph, t_extract)

    t0 = time.monotonic()
    n = complete_graph(graph, TEXT, backend(MID, temperature=0))
    t_complete = time.monotonic() - t0
    record("coder-next/+complétude", graph, t_extract + t_complete,
           detail=f"{n} items proposés en {t_complete:.1f}s")

    t0 = time.monotonic()
    before = len(graph.entities)
    graph = canonicalize_graph(graph, backend(MID, temperature=0))
    t_canon = time.monotonic() - t0
    record("coder-next/+complétude+canon", graph, t_extract + t_complete + t_canon,
           detail=f"entités {before}->{len(graph.entities)} en {t_canon:.1f}s")

    # --- Voie bruit : union des 3 petits + canonicalisation -------------------
    warmup(SMALLS)
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SMALLS)) as pool:
        futures = {m: pool.submit(extract_graph, TEXT, backend(m), CHUNK_SIZE) for m in SMALLS}
        graphs = {m: f.result() for m, f in futures.items()}
    union_small = merge_union([graphs[m] for m in SMALLS])
    t_extract = time.monotonic() - t0
    record("union3petits/seule", union_small, t_extract)

    t0 = time.monotonic()
    before = len(union_small.entities)
    union_small = canonicalize_graph(union_small, backend("gpt-oss:20b", temperature=0))
    t_canon = time.monotonic() - t0
    record("union3petits/+canon", union_small, t_extract + t_canon,
           detail=f"entités {before}->{len(union_small.entities)} en {t_canon:.1f}s")

    # --- Voie exactitude : union(120b, coder-next) + canonicalisation ---------
    t0 = time.monotonic()
    graphs_big = [extract_graph(TEXT, backend(m), CHUNK_SIZE) for m in (BIG, MID)]
    union_big = merge_union(graphs_big)
    t_extract = time.monotonic() - t0
    record("union2gros/seule", union_big, t_extract)

    t0 = time.monotonic()
    before = len(union_big.entities)
    union_big = canonicalize_graph(union_big, backend(MID, temperature=0))
    t_canon = time.monotonic() - t0
    record("union2gros/+canon", union_big, t_extract + t_canon,
           detail=f"entités {before}->{len(union_big.entities)} en {t_canon:.1f}s")

    (out_dir / "refine_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TERMINÉ", flush=True)


if __name__ == "__main__":
    main()
