"""Handlers purs (testables) + adaptateur http.server mince pour l'atelier."""
from __future__ import annotations

import json
import sqlite3
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .metrics import bcubed, confusion_pairs, lea
from .reader import context_for_entity, predicted_partition, surface_forms
from .store import GoldStore

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def build_candidates(conn: sqlite3.Connection, store: GoldStore, chunks: list) -> list:
    ann = {(a.entity_id, a.kind, a.surface_form): a for a in store.all()}
    pred = predicted_partition(conn)
    ctx_cache: dict = {}
    out = []
    for s in surface_forms(conn):
        if s.entity_id not in ctx_cache:
            ctx_cache[s.entity_id] = context_for_entity(conn, s.entity_id, chunks)
        a = ann.get((s.entity_id, s.kind, s.surface_form))
        out.append({
            "entity_id": s.entity_id, "kind": s.kind, "surface_form": s.surface_form,
            "entity_type": s.entity_type,
            "predicted_cluster": pred[(s.entity_id, s.kind, s.surface_form)],
            "context": ctx_cache[s.entity_id],
            "annotation": {
                "referent_id": a.referent_id if a else None,
                "referent_type": a.referent_type if a else None,
                "discarded": a.discarded if a else False,
            },
        })
    return out


def apply_annotation(store: GoldStore, payload: dict) -> None:
    store.upsert(
        entity_id=payload["entity_id"], kind=payload["kind"],
        surface_form=payload["surface_form"],
        referent_id=payload.get("referent_id"),
        referent_type=payload.get("referent_type"),
        discarded=payload.get("discarded", False),
    )


def compute_score(conn: sqlite3.Connection, store: GoldStore) -> dict:
    pred = predicted_partition(conn)
    gold = store.gold_partition()
    bp, br, bf = bcubed(pred, gold)
    lp, lr, lf = lea(pred, gold)
    over, under = confusion_pairs(pred, gold)
    n_discarded = sum(1 for a in store.all() if a.discarded)
    return {
        "bcubed": {"p": bp, "r": br, "f": bf},
        "lea": {"p": lp, "r": lr, "f": lf},
        "over_merged": over, "under_merged": under,
        "n_evaluated": len([k for k in pred if k in gold]),
        "n_discarded": n_discarded,
    }


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, conn, store, chunks, **kwargs):
        self._conn, self._store, self._chunks = conn, store, chunks
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):  # silence les logs http par défaut
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (WEB_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/candidates":
            self._send_json({"candidates": build_candidates(self._conn, self._store, self._chunks)})
        elif self.path == "/api/score":
            self._send_json(compute_score(self._conn, self._store))
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            apply_annotation(self._store, payload)
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)


def serve(conn, store, chunks, host="127.0.0.1", port=8000):
    handler = partial(_Handler, conn=conn, store=store, chunks=chunks)
    httpd = HTTPServer((host, port), handler)
    print(f"annota sur http://{host}:{port}")
    httpd.serve_forever()
