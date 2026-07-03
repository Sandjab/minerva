"""Bench des stratégies d'ensemble : exactitude (2 gros, séquentiel) et
efficacité (3 petits en parallèle vs 1 moyen).

Usage :
    .venv/bin/python benchmarks/bench_ensemble.py

Écrit les résultats dans benchmarks/results/<date>/ensemble_results.json.
Le chrono mural inclut l'extraction (parallèle quand les modèles cohabitent
en mémoire, séquentielle sinon) et l'éventuel appel d'arbitrage.
"""

import concurrent.futures
import datetime
import importlib.util
import json
import time
from pathlib import Path

from minerva.ensemble import merge_arbitrated, merge_union, merge_vote
from minerva.extraction import extract_graph
from minerva.llm.openai_backend import OpenAIBackend

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
TEXT, CHUNK_SIZE, score = _bench.TEXT, _bench.CHUNK_SIZE, _bench.score

OLLAMA = "http://localhost:11434/v1"

BIG_A = "gpt-oss:120b"
BIG_B = "qwen3-coder-next:latest"
SMALLS = ["gpt-oss:20b", "gemma4:latest", "qwen2.5:7b-instruct"]  # ~25 Go cumulés
ARBITER_BIG = "qwen3-coder-next:latest"
ARBITER_SMALL = "gpt-oss:20b"
MID_REFERENCE = "qwen3-coder-next:latest"


def backend(model: str, temperature: float | None = None) -> OpenAIBackend:
    return OpenAIBackend(model=model, base_url=OLLAMA, temperature=temperature)


def warmup(models: list[str]) -> None:
    for m in models:
        backend(m)._client.chat.completions.create(
            model=m, messages=[{"role": "user", "content": "Réponds : ok"}]
        )


def extract_all(models: list[str], parallel: bool) -> tuple[dict, float]:
    """Extrait le graphe avec chaque modèle ; renvoie (graphes, chrono mural)."""
    t0 = time.monotonic()
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {m: pool.submit(extract_graph, TEXT, backend(m), CHUNK_SIZE) for m in models}
            graphs = {m: f.result() for m, f in futures.items()}
    else:
        graphs = {m: extract_graph(TEXT, backend(m), CHUNK_SIZE) for m in models}
    return graphs, time.monotonic() - t0


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
        (out_dir / f"ensemble_{safe}.json").write_text(
            json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Voie exactitude : 2 gros modèles, séquentiel (ne cohabitent pas) ----
    print("=== exactitude : extractions des 2 gros (séquentiel) ===", flush=True)
    graphs_big, t_big = extract_all([BIG_A, BIG_B], parallel=False)
    ordered_big = [graphs_big[BIG_A], graphs_big[BIG_B]]  # 120b prioritaire

    record("exactitude/union", merge_union(ordered_big), t_big)
    record("exactitude/vote2", merge_vote(ordered_big, min_votes=2), t_big)

    t0 = time.monotonic()
    merged = merge_arbitrated(
        ordered_big, backend(ARBITER_BIG, temperature=0), TEXT, min_votes=2
    )
    record("exactitude/arbitré", merged, t_big + (time.monotonic() - t0),
           detail=f"arbitre={ARBITER_BIG}")

    # --- Voie efficacité : 3 petits en parallèle vs 1 moyen -------------------
    print("=== efficacité : chargement des 3 petits ===", flush=True)
    warmup(SMALLS)
    graphs_small, t_small = extract_all(SMALLS, parallel=True)
    ordered_small = [graphs_small[m] for m in SMALLS]  # 20b prioritaire

    record("efficacité/majorité2sur3", merge_vote(ordered_small, min_votes=2), t_small)
    record("efficacité/union3", merge_union(ordered_small), t_small)

    t0 = time.monotonic()
    merged = merge_arbitrated(
        ordered_small, backend(ARBITER_SMALL, temperature=0), TEXT, min_votes=2
    )
    record("efficacité/arbitré", merged, t_small + (time.monotonic() - t0),
           detail=f"arbitre={ARBITER_SMALL}")

    # Référence : le meilleur modèle "moyen" seul, même chrono mural
    warmup([MID_REFERENCE])
    graphs_ref, t_ref = extract_all([MID_REFERENCE], parallel=False)
    record("référence/coder-next_seul", graphs_ref[MID_REFERENCE], t_ref)

    (out_dir / "ensemble_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TERMINÉ", flush=True)


if __name__ == "__main__":
    main()
