"""CLI annota : `annota serve` (atelier) et `annota score` (métriques)."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .reader import build_chunks
from .server import compute_score, serve
from .store import GoldStore


def _open(base: str, gold: str, source: str | None, chunk_size: int):
    conn = sqlite3.connect(base)
    gpath = Path(gold)
    store = GoldStore.open(gpath) if gpath.exists() else GoldStore.create(gpath, source_db=base)
    chunks = build_chunks(Path(source).read_text(encoding="utf-8"), chunk_size) if source else []
    return conn, store, chunks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="annota")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("serve", help="lancer l'atelier d'annotation")
    ps.add_argument("base")
    ps.add_argument("--source", required=True)
    ps.add_argument("--gold", default="gold.sqlite")
    ps.add_argument("--chunk-size", type=int, default=8000)
    ps.add_argument("--port", type=int, default=8000)

    pc = sub.add_parser("score", help="mesurer canon_alias contre le gold")
    pc.add_argument("base")
    pc.add_argument("--gold", default="gold.sqlite")

    args = p.parse_args(argv)
    if args.cmd == "serve":
        conn, store, chunks = _open(args.base, args.gold, args.source, args.chunk_size)
        serve(conn, store, chunks, port=args.port)
        return 0
    if args.cmd == "score":
        conn = sqlite3.connect(args.base)
        store = GoldStore.open(Path(args.gold))
        print(json.dumps(compute_score(conn, store), ensure_ascii=False, indent=2))
        return 0
    return 1
