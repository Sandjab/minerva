"""Re-score les graphes sauvegardés d'un bench avec le pipeline minerva courant.

Les benchs sauvegardent le graphe JSON de chaque run ; quand le pipeline
évolue (normalisation, repli d'alias, rejet du bruit…), ce script recharge
ces graphes à travers le code actuel et recalcule les scores — sans relancer
les LLM. Usage :

    .venv/bin/python benchmarks/rescore.py results/2026-07-03

Compare aux scores enregistrés dans bench_results.json et écrit
bench_results_rescored.json (l'original n'est jamais modifié).
"""

import importlib.util
import json
import sys
from pathlib import Path

from minerva.model import Entity, KnowledgeGraph, Relation

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location("bench", HERE / "bench.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
score = _bench.score


def load_graph(path: Path) -> tuple[KnowledgeGraph, int]:
    """Recharge un graphe sauvegardé via le pipeline courant.

    Les enregistrements invalides au sens des règles actuelles (noms vides…)
    sont écartés, comme le ferait extraction.sanitize sur une sortie LLM.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = KnowledgeGraph()
    skipped = 0
    for e in data["entities"]:
        try:
            graph.add_entity(Entity(**e))
        except ValueError:
            skipped += 1
    for r in data["relations"]:
        try:
            graph.add_relation(Relation(**r))
        except ValueError:
            skipped += 1
    return graph, skipped


def main(results_dir: Path) -> None:
    results_path = results_dir / "bench_results.json"
    recorded = {
        e["model"]: e
        for e in json.loads(results_path.read_text(encoding="utf-8"))
        # les entrées agrégées multi-runs n'ont pas un graphe unique à re-scorer
        if "error" not in e and "per_run" not in e
    }

    rescored = []
    for model, old in recorded.items():
        safe = model.replace(":", "_").replace("/", "_")
        graph_path = results_dir / f"bench_{safe}.json"
        if not graph_path.exists():
            print(f"!! graphe absent pour {model} ({graph_path.name})")
            continue
        graph, skipped = load_graph(graph_path)
        new = {"model": model, "load_s": old.get("load_s"), "time_s": old.get("time_s")}
        new.update(score(graph))
        new["skipped_records"] = skipped
        rescored.append(new)

        diffs = [
            f"{key}: {old[key]} -> {new[key]}"
            for key in (
                "n_entities", "n_relations", "expected_entities",
                "expected_relations", "valjean_madeleine_merged",
            )
            if old.get(key) != new.get(key)
        ]
        print(f"== {model}")
        print("   " + ("; ".join(diffs) if diffs else "inchangé"))

    out = results_dir / "bench_results_rescored.json"
    out.write_text(json.dumps(rescored, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    rel = sys.argv[1] if len(sys.argv) > 1 else "results/2026-07-03"
    main((HERE / rel).resolve())
