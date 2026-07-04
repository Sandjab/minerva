"""Bench du volet timeline : texte inédit à structure temporelle contrôlée
(flashback, ellipse quantifiée, fil parallèle), vérité terrain sur l'ordre
diégétique des âges d'Élise (12 < 30 < 50) alors que l'ordre de LECTURE est
30, 12, 50 — c'est la désynchronisation que le pipeline doit capturer.

Usage :
    .venv/bin/python benchmarks/bench_timeline.py [modèle ...] [--runs N]
Prérequis : serveur Ollama local sur http://localhost:11434.
"""

import datetime
import json
import re
import time
import traceback
from pathlib import Path

from minerva.extraction import extract_graph
from minerva.llm.openai_backend import OpenAIBackend
from minerva.model import KnowledgeGraph
from minerva.timeline import AVANT, SIMULTANE

HERE = Path(__file__).parent
TEXT = (HERE / "timeline_texte.txt").read_text(encoding="utf-8")
CHUNK_SIZE = 700  # force plusieurs chunks : les scènes traversent les chunks

DEFAULT_MODELS = ["gpt-oss:120b", "qwen3-coder-next:latest"]

EXPECTED_ENTITIES = {
    "Élise Chardon": ["Élise Chardon", "Élise"],
    "Bastien Malot": ["Bastien Malot", "Bastien"],
    "Aurélien Chardon": ["Aurélien Chardon", "Aurélien"],
    "Camille Roche": ["Camille Roche", "Camille"],
    "Aubervilliers": ["Aubervilliers"],
    "Mirecourt": ["Mirecourt"],
    "Lyon": ["Lyon"],
}

_NUMBER_WORDS = {
    "douze": 12, "seize": 16, "trente": 30, "cinquante": 50,
}
EXPECTED_AGES = (12, 30, 50)  # Élise, en ordre diégétique
ELLIPSE_DAYS = 20 * 365


def parse_age(value: str) -> int | None:
    value = value.casefold()
    match = re.search(r"\d+", value)
    if match:
        return int(match.group())
    for word, n in _NUMBER_WORDS.items():
        # word.rstrip("e") tolère les formes approximatives françaises en
        # "-aine" (« la cinquantaine ») : « cinquante » n'est pas une
        # sous-chaîne de « cinquantaine », mais « cinquant » l'est.
        if word.rstrip("e") in value:
            return n
    return None


def resolve_any(graph: KnowledgeGraph, variants: list[str]):
    for v in variants:
        e = graph.resolve(v)
        if e is not None:
            return e
    return None


def score(graph: KnowledgeGraph) -> dict:
    graph.timeline.resolve()
    entities_found = {
        name: resolve_any(graph, variants) is not None
        for name, variants in EXPECTED_ENTITIES.items()
    }

    elise = resolve_any(graph, EXPECTED_ENTITIES["Élise Chardon"])
    order = {m.id: m.resolved_order for m in graph.timeline.moments}
    days = {m.id: m.resolved_days for m in graph.timeline.moments}
    reading = {m.id: (m.chunk_index, m.seq) for m in graph.timeline.moments}

    # (âge, ordre diégétique, ordre de lecture, jours, moment_id) des constats
    # d'âge d'Élise
    ages: dict[int, tuple] = {}
    if elise is not None:
        for a in graph.assertions:
            if a.entity != elise.name or "âge" not in a.attribute.casefold():
                continue
            age = parse_age(a.value)
            if age in EXPECTED_AGES and a.moment_id is not None and age not in ages:
                ages[age] = (order.get(a.moment_id), reading.get(a.moment_id),
                             days.get(a.moment_id), a.moment_id)

    ages_captured = sorted(ages)
    diegetic_ok = (
        len(ages) == 3
        and all(ages[a][0] is not None for a in EXPECTED_AGES)
        and ages[12][0] < ages[30][0] < ages[50][0]
    )
    # Contrôle que le test mesure bien quelque chose : en LECTURE, 12 vient après 30.
    reading_disorder = (
        12 in ages and 30 in ages and ages[12][1] is not None
        and ages[30][1] is not None and ages[12][1] > ages[30][1]
    )
    # L'ellipse se mesure sur la CONTRAINTE quantifiée qui mène au moment de
    # l'âge 50 (ou à un moment simultané), pas sur la coordonnée absolue :
    # resolved_days s'ancre au premier moment résolu (souvent le flashback),
    # qui n'a pas toujours de chemin quantifié vers l'ellipse — la coordonnée
    # serait alors None sans que l'extraction soit fautive.
    ellipse_days = None
    if 50 in ages:
        fifty_mid = ages[50][3]
        group = {fifty_mid}
        for c in graph.timeline.constraints:
            if c.relation == SIMULTANE:
                if c.target_id == fifty_mid:
                    group.add(c.source_id)
                elif c.source_id == fifty_mid:
                    group.add(c.target_id)
        for c in graph.timeline.constraints:
            if c.relation == AVANT and c.gap.days is not None and c.target_id in group:
                ellipse_days = c.gap.days
                break
    ellipse_ok = (
        ellipse_days is not None
        and abs(ellipse_days - ELLIPSE_DAYS) / ELLIPSE_DAYS <= 0.25
    )

    return {
        "n_moments": len(graph.timeline.moments),
        "n_constraints": len(graph.timeline.constraints),
        "n_assertions": len(graph.assertions),
        "expected_entities": sum(entities_found.values()),
        "expected_entities_total": len(entities_found),
        "missing_entities": [n for n, ok in entities_found.items() if not ok],
        "ages_captured": ages_captured,
        "diegetic_order_ok": diegetic_ok,
        "reading_disorder_present": reading_disorder,
        "ellipse_days": ellipse_days,
        "ellipse_ok": ellipse_ok,
    }


def run_model(model: str, out_dir: Path, runs: int = 1) -> dict:
    entry: dict = {"model": model, "runs": runs}
    backend = OpenAIBackend(model=model, base_url="http://localhost:11434/v1")
    t0 = time.monotonic()
    backend._client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Réponds : ok"}]
    )
    entry["load_s"] = round(time.monotonic() - t0, 1)

    safe = model.replace(":", "_").replace("/", "_")
    per_run: list[dict] = []
    for i in range(runs):
        t0 = time.monotonic()
        graph = extract_graph(
            TEXT, backend, chunk_size=CHUNK_SIZE,
            on_progress=lambda d, t: print(f"  run {i + 1}/{runs} chunk {d}/{t}", flush=True),
        )
        run_entry = {"time_s": round(time.monotonic() - t0, 1)}
        run_entry.update(score(graph))
        per_run.append(run_entry)
        suffix = f"_run{i + 1}" if runs > 1 else ""
        (out_dir / f"timeline_{safe}{suffix}.json").write_text(
            json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if runs == 1:
        entry.update(per_run[0])
    else:
        entry["per_run"] = per_run
        for key in ("diegetic_order_ok", "ellipse_ok"):
            entry[f"{key}_rate"] = f"{sum(1 for r in per_run if r[key])}/{runs}"
        entry["n_moments_values"] = [r["n_moments"] for r in per_run]
    return entry


def main(models: list[str], runs: int = 1) -> None:
    out_dir = HERE / "results" / datetime.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "bench_timeline_results.json"
    previous: list[dict] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    )
    new_keys = {(m, runs) for m in models}
    results = [e for e in previous if (e.get("model"), e.get("runs", 1)) not in new_keys]
    for model in models:
        print(f"=== {model} (runs={runs}) ===", flush=True)
        try:
            entry = run_model(model, out_dir, runs=runs)
        except Exception as exc:
            entry = {"model": model, "runs": runs, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TERMINÉ -> {out_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bench timeline minerva sur Ollama")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    main(args.models or DEFAULT_MODELS, runs=args.runs)
