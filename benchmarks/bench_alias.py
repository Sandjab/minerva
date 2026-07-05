"""Bench isolé de la passe d'alias / identité d'emprunt (resolve_aliases).

Applique la passe sur les graphes « actée » DÉJÀ sauvegardés d'une campagne de
référence (mêmes entrées que la baseline actée) et rescore. Isole ainsi l'effet
pur de la passe — sans rejouer l'extraction, donc sans sa variance — et le rend
directement comparable au baseline. Un seul appel LLM par (graphe × portée),
température 0.

La passe ne dépend que du graphe et du texte ; l'appliquer aujourd'hui à des
graphes produits par un pipeline antérieur est donc licite.

Usage :
    .venv/bin/python benchmarks/bench_alias.py [--baseline results/2026-07-05] \
        [--model gpt-oss:120b ...] [--scope impersonation broad]
Prérequis : serveur Ollama local, graphes ref_reference_actee_*.json présents.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

from minerva.llm.openai_backend import OpenAIBackend
from minerva.refine import resolve_aliases

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
MODELS = ["gpt-oss:120b", "qwen3-coder-next:latest"]
SCOPES = ["impersonation", "broad"]
# fusion difficile scorée à part : l'identité d'emprunt Sérac = Rivière
IMPERSONATION_MERGE = [["Antoine Sérac", "Sérac"], ["Théo Rivière", "Rivière"]]


def backend(model: str) -> OpenAIBackend:
    return OpenAIBackend(model=model, base_url=OLLAMA, temperature=0)


def impersonation_ok(score: dict) -> bool:
    """Vrai si la fusion d'identité d'emprunt est réussie dans ce score."""
    return IMPERSONATION_MERGE not in score["failed_merges"]


def summarize(score: dict) -> dict:
    return {
        "merge_rate": score["merge_rate"],
        "impersonation": impersonation_ok(score),
        "entity_precision": score["entity_precision"],
        "relation_precision": score["relation_precision"],
        "n_entities": score["n_entities"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="results/2026-07-05")
    parser.add_argument("--model", nargs="*", default=MODELS)
    parser.add_argument("--scope", nargs="*", default=SCOPES)
    args = parser.parse_args()

    base_dir = (HERE / args.baseline).resolve()
    out_dir = base_dir / "alias"
    out_dir.mkdir(exist_ok=True)
    text = (HERE / "reference_texte.txt").read_text(encoding="utf-8")
    ref = _scoring.load_reference(HERE / "reference_reference_texte.json")

    rows: list[dict] = []
    # agrégats {(model, scope): {merges_ok, merges_total, imp_ok, imp_n, eP[], rP[]}}
    agg: dict[tuple[str, str], dict] = {}

    for model in args.model:
        safe = model.replace(":", "_").replace("/", "_")
        graphs = sorted(base_dir.glob(f"ref_reference_actee_{safe}_run*.json"))
        if not graphs:
            print(f"!! aucun graphe actée pour {model} ({safe})", flush=True)
            continue
        # échauffement : charge le modèle hors mesure
        backend(model)._client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Réponds : ok"}]
        )
        for gp in graphs:
            base_graph, _ = _rescore.load_graph(gp)
            base_score = _scoring.score_reference(base_graph, ref)
            row = {"graph": gp.name, "model": model,
                   "baseline": summarize(base_score)}
            for scope in args.scope:
                graph, _ = _rescore.load_graph(gp)  # recharge propre par portée
                merged = resolve_aliases(graph, text, backend(model), scope=scope)
                score = _scoring.score_reference(merged, ref)
                row[scope] = summarize(score)
                (out_dir / f"ref_reference_alias-{scope}_{safe}_{gp.stem.split('_')[-1]}.json"
                 ).write_text(json.dumps(merged.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")
                a = agg.setdefault((model, scope),
                                   {"ok": 0, "tot": 0, "imp_ok": 0, "imp_n": 0,
                                    "eP": [], "rP": []})
                a["ok"] += score["merges_ok"]
                a["tot"] += score["merges_total"]
                a["imp_ok"] += int(impersonation_ok(score))
                a["imp_n"] += 1
                a["eP"].append(score["entity_precision"])
                a["rP"].append(score["relation_precision"])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = []
    print("\n=== agrégats (modèle × portée) ===", flush=True)
    print(f"{'modèle':22} {'portée':14} {'fusions':>8} {'Sérac=Riv':>10} "
          f"{'eP':>6} {'rP':>6}", flush=True)
    for (model, scope), a in sorted(agg.items()):
        eP = round(statistics.mean(a["eP"]), 3)
        rP = round(statistics.mean(a["rP"]), 3)
        line = {"model": model, "scope": scope,
                "merge_rate": f"{a['ok']}/{a['tot']}",
                "impersonation_rate": f"{a['imp_ok']}/{a['imp_n']}",
                "entity_precision_mean": eP, "relation_precision_mean": rP}
        summary.append(line)
        print(f"{model:22} {scope:14} {a['ok']}/{a['tot']:<6} "
              f"{a['imp_ok']}/{a['imp_n']:<8} {eP:>6} {rP:>6}", flush=True)

    out = out_dir / "alias_results.json"
    out.write_text(json.dumps({"rows": rows, "summary": summary},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTERMINÉ -> {out}", flush=True)


if __name__ == "__main__":
    main()
