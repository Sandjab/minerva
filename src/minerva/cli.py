"""CLI : minerva extract | show | export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import store
from .chunking import DEFAULT_CHUNK_SIZE
from .extraction import extract_graph
from .llm import make_backend
from .model import KnowledgeGraph


def _cmd_extract(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8")
    backend = make_backend(
        args.provider, model=args.model, base_url=args.base_url, temperature=args.temperature
    )

    def progress(done: int, total: int) -> None:
        print(f"chunk {done}/{total}", file=sys.stderr)

    graph = extract_graph(text, backend, chunk_size=args.chunk_size, on_progress=progress)
    store.save(graph, args.output)
    print(
        f"{len(graph.entities)} entités, {len(graph.relations)} relations, "
        f"{len(graph.timeline.moments)} moments -> {args.output}"
    )
    return 0


def _format_entity(graph: KnowledgeGraph, name: str) -> str:
    entity = graph.resolve(name)
    if entity is None:
        return f"Entité introuvable : {name}"
    graph.timeline.resolve()
    lines = [f"{entity.name} ({entity.type})"]
    if entity.aliases:
        lines.append(f"  alias : {', '.join(entity.aliases)}")
    history = _attribute_history(graph, entity.name)
    for attr, value in sorted(entity.attributes.items()):
        lines.append(f"  {attr} : {value}")
        if len(history.get(attr, [])) > 1:
            lines.append(f"    historique : {' → '.join(history[attr])}")
    related = [r for r in graph.relations if entity.name in (r.source, r.target)]
    for rel in related:
        attrs = f" [{', '.join(f'{k}={v}' for k, v in rel.attributes.items())}]" if rel.attributes else ""
        lines.append(f"  {rel.source} --{rel.name}--> {rel.target}{attrs}")
    return "\n".join(lines)


def _attribute_history(graph: KnowledgeGraph, entity_name: str) -> dict[str, list[str]]:
    """Valeurs successives par attribut, en ordre diégétique résolu ; les
    doublons consécutifs sont fusionnés."""
    order = {m.id: m.resolved_order or 0 for m in graph.timeline.moments}
    dated = sorted(
        (a for a in graph.assertions
         if a.entity == entity_name and a.attribute and a.moment_id is not None),
        key=lambda a: order.get(a.moment_id, 0),
    )
    history: dict[str, list[str]] = {}
    for a in dated:
        values = history.setdefault(a.attribute, [])
        if not values or values[-1] != a.value:
            values.append(a.value)
    return history


def _format_timeline(graph: KnowledgeGraph) -> str:
    graph.timeline.resolve()
    moments = sorted(graph.timeline.moments, key=lambda m: m.resolved_order or 0)
    if not moments:
        return "aucun moment narratif enregistré (graphe legacy ou extraction sans timeline)"
    lines = []
    for m in moments:
        days = f" · jour {m.resolved_days:.0f}" if m.resolved_days is not None else ""
        lines.append(f"M{m.id} · ordre {m.resolved_order} · chunk {m.chunk_index}{days} — {m.summary}")
    lines.append(f"{len(moments)} moments, {len(graph.timeline.constraints)} contraintes")
    return "\n".join(lines)


def _cmd_timeline(args: argparse.Namespace) -> int:
    print(_format_timeline(store.load(args.database)))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    graph = store.load(args.database)
    if args.entity:
        print(_format_entity(graph, args.entity))
        return 0
    print(f"{len(graph.entities)} entités, {len(graph.relations)} relations")
    for entity in sorted(graph.entities, key=lambda e: e.name.lower()):
        print(f"  {entity.name} ({entity.type}) — {len(entity.attributes)} attribut(s)")
    for rel in graph.relations:
        print(f"  {rel.source} --{rel.name}--> {rel.target}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    graph = store.load(args.database)
    Path(args.output).write_text(
        json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"exporté -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minerva", description="Extraction d'entités et de relations d'un texte")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="extraire le graphe d'un fichier texte")
    p_extract.add_argument("input", help="fichier texte à analyser")
    p_extract.add_argument("-o", "--output", required=True, help="base SQLite de sortie")
    p_extract.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p_extract.add_argument("--model", default=None, help="identifiant du modèle")
    p_extract.add_argument("--base-url", default=None, help="URL d'un serveur compatible OpenAI (ex. http://localhost:11434/v1)")
    p_extract.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_extract.add_argument("--temperature", type=float, default=None,
                           help="température d'échantillonnage (backend openai/Ollama uniquement)")
    p_extract.set_defaults(func=_cmd_extract)

    p_show = sub.add_parser("show", help="afficher le contenu d'une base")
    p_show.add_argument("database", help="base SQLite")
    p_show.add_argument("--entity", default=None, help="détail d'une entité (nom ou alias)")
    p_show.set_defaults(func=_cmd_show)

    p_export = sub.add_parser("export", help="exporter une base en JSON")
    p_export.add_argument("database", help="base SQLite")
    p_export.add_argument("-o", "--output", required=True, help="fichier JSON de sortie")
    p_export.set_defaults(func=_cmd_export)

    p_timeline = sub.add_parser("timeline", help="afficher les moments narratifs résolus")
    p_timeline.add_argument("database", help="base SQLite")
    p_timeline.set_defaults(func=_cmd_timeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
