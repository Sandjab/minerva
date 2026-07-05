"""Révision de la reco pipeline (120b) — addendum 9.

À extraction FIXÉE (graphes « nue » sauvegardés = les extractions de la
campagne), rejoue les combinaisons de raffinement et les score contre la
référence exhaustive courante. Isole l'effet du raffinement : même extraction
de base, pas de ré-extraction ni de variance, temps de raffinement mesuré. Les
lignes `nue`/`actee` recalculées servent de contrôle vis-à-vis du rescore
(addendum 7).

Pipelines (le graphe nue = l'extraction, ~185 s/run d'après l'addendum 7) :
  nue          extraction seule
  actee        + complétude + canonicalisation
  canon_alias  + canonicalisation + alias   (SANS complétude)
  actee_alias  + complétude + canonicalisation + alias

Usage :
    .venv/bin/python benchmarks/bench_pipeline_reco.py \
        [--model gpt-oss:120b] [--baseline results/2026-07-05]
Prérequis : Ollama local, graphes ref_reference_nue_<model>_run*.json présents.
"""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

from minerva.llm.openai_backend import OpenAIBackend
from minerva.refine import canonicalize_graph, complete_graph, resolve_aliases

HERE = Path(__file__).parent

_spec = importlib.util.spec_from_file_location(
    "reference_scoring", HERE / "reference_scoring.py"
)
_scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring)

_spec_r = importlib.util.spec_from_file_location("rescore", HERE / "rescore.py")
_rescore = importlib.util.module_from_spec(_spec_r)
_spec_r.loader.exec_module(_rescore)

OLLAMA = "http://localhost:11434/v1"
PIPELINES = ["nue", "actee", "canon_alias", "actee_alias"]
IMPERSONATION_MERGE = [["Antoine Sérac", "Sérac"], ["Théo Rivière", "Rivière"]]


def backend(model: str) -> OpenAIBackend:
    return OpenAIBackend(model=model, base_url=OLLAMA, temperature=0)


def apply_pipeline(nue_path: Path, pipeline: str, text: str, model: str):
    """Recharge l'extraction nue et applique le raffinement du pipeline.
    Renvoie (graphe, secondes de raffinement)."""
    graph, _ = _rescore.load_graph(nue_path)
    t0 = time.monotonic()
    if pipeline in ("actee", "actee_alias"):
        complete_graph(graph, text, backend(model))
    if pipeline in ("actee", "canon_alias", "actee_alias"):
        graph = canonicalize_graph(graph, backend(model))
    if pipeline in ("canon_alias", "actee_alias"):
        graph = resolve_aliases(graph, text, backend(model))
    return graph, time.monotonic() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument("--baseline", default="results/2026-07-05")
    args = parser.parse_args()
    model = args.model
    safe = model.replace(":", "_").replace("/", "_")

    base_dir = (HERE / args.baseline).resolve()
    out_dir = base_dir / "reco"
    out_dir.mkdir(exist_ok=True)
    text = (HERE / "reference_texte.txt").read_text(encoding="utf-8")
    ref = _scoring.load_reference(HERE / "reference_reference_texte.json")
    nue_graphs = sorted(base_dir.glob(f"ref_reference_nue_{safe}_run*.json"))
    if not nue_graphs:
        raise SystemExit(f"aucun graphe nue pour {model} ({safe}) dans {base_dir}")

    # échauffement hors mesure
    backend(model)._client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Réponds : ok"}]
    )

    rows: list[dict] = []
    agg: dict[str, dict] = {}
    for pipeline in PIPELINES:
        for gp in nue_graphs:
            graph, refine_s = apply_pipeline(gp, pipeline, text, model)
            s = _scoring.score_reference(graph, ref)
            imp = IMPERSONATION_MERGE not in s["failed_merges"]
            row = {"pipeline": pipeline, "graph": gp.name, "refine_s": round(refine_s, 1),
                   "n_entities": s["n_entities"], "impersonation": imp,
                   "entity_precision": s["entity_precision"], "entity_recall": s["entity_recall"],
                   "relation_precision": s["relation_precision"], "relation_recall": s["relation_recall"],
                   "merge_rate": s["merge_rate"]}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            (out_dir / f"ref_reference_{pipeline}_{safe}_{gp.stem.split('_')[-1]}.json"
             ).write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
            a = agg.setdefault(pipeline, {k: [] for k in
                ("eP", "eR", "rP", "rR", "refine_s", "n_ent")})
            a.setdefault("ok", 0); a.setdefault("tot", 0); a.setdefault("imp", 0)
            a["eP"].append(s["entity_precision"]); a["eR"].append(s["entity_recall"])
            a["rP"].append(s["relation_precision"]); a["rR"].append(s["relation_recall"])
            a["refine_s"].append(refine_s); a["n_ent"].append(s["n_entities"])
            a["ok"] += s["merges_ok"]; a["tot"] += s["merges_total"]; a["imp"] += int(imp)

    def mean(xs): return round(statistics.mean(xs), 3)

    summary = []
    print("\n=== agrégats %s / reference (extraction ~185 s/run) ===" % model, flush=True)
    print("%-12s %6s %6s %6s %6s %8s %9s %9s" % (
        "pipeline", "eP", "eR", "rP", "rR", "fusions", "Sérac=Riv", "raffin.s"), flush=True)
    for p in PIPELINES:
        a = agg[p]
        line = {"pipeline": p, "entity_precision": mean(a["eP"]), "entity_recall": mean(a["eR"]),
                "relation_precision": mean(a["rP"]), "relation_recall": mean(a["rR"]),
                "merge_rate": "%d/%d" % (a["ok"], a["tot"]),
                "impersonation_rate": "%d/%d" % (a["imp"], len(nue_graphs)),
                "refine_s_mean": round(statistics.mean(a["refine_s"]), 1),
                "n_entities_mean": mean(a["n_ent"])}
        summary.append(line)
        print("%-12s %6.3f %6.3f %6.3f %6.3f %8s %9s %8.1f" % (
            p, line["entity_precision"], line["entity_recall"], line["relation_precision"],
            line["relation_recall"], line["merge_rate"], line["impersonation_rate"],
            line["refine_s_mean"]), flush=True)

    out = out_dir / "reco_results.json"
    out.write_text(json.dumps({"model": model, "rows": rows, "summary": summary},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nTERMINÉ -> %s" % out, flush=True)


if __name__ == "__main__":
    main()
