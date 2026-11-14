"""Day 06 pipeline entry point.

Run from starter/:
  python -m src.prepare             # read/normalize PDF -> pages.jsonl, manifest, chunk stats
  python -m src.verify_benchmark    # check benchmark evidence BEFORE retrieval
  python -m src.run ingest          # embed + upsert A/B/C into Chroma (idempotent)
  python -m src.run query --split dev
  python -m src.run query --split test   # only after the dev-based choice is fixed
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

from . import config
from .embeddings import E5Embedder
from .evaluation import evaluate_query, summarize
from .prepare import build_chunks, length_stats, load_pages
from .store import ChromaStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{pkg: version(pkg) for pkg in ("chromadb", "sentence-transformers", "torch", "pypdfium2", "pypdf")}}


def ingest(store: ChromaStore, embedder: E5Embedder) -> dict:
    pages = load_pages()
    if len(pages) < 30:
        raise SystemExit("fewer than 30 content pages")
    report = {"run_at": _now(), "model": embedder.model_name, "dimensions": embedder.dimensions,
              "device": embedder.device, "env": _env(), "strategies": {}}
    b_vectors: dict[str, list[float]] = {}
    for strategy in ("A", "B", "C"):
        t0 = time.perf_counter()
        chunks = build_chunks(pages, strategy)
        t_chunk = time.perf_counter() - t0
        before = store.count(strategy)
        existing = store.existing(strategy, config.DOCUMENT_ID)
        new_ids = {c.chunk_id for c in chunks}
        stale = [i for i in existing if i not in new_ids]          # chunks of an older version
        todo = [c for c in chunks if existing.get(c.chunk_id) != c.metadata["content_sha1"]]

        t1 = time.perf_counter()
        if strategy == "C":
            # C uses exactly B's chunks; reuse B's vectors so B and C differ only by the filter.
            vecs = [b_vectors[c.chunk_id.replace("C-", "B-", 1)] for c in todo]
        else:
            vecs = embedder.embed_documents([c.text for c in todo])
        t_embed = time.perf_counter() - t1

        t2 = time.perf_counter()
        store.delete(strategy, stale)
        store.upsert(strategy, todo, vecs)
        t_store = time.perf_counter() - t2
        after = store.count(strategy)
        if strategy == "B":
            b_vectors = store.get_vectors("B", [c.chunk_id for c in chunks])
        tokens = [embedder.count_tokens(c.text) for c in chunks]
        report["strategies"][strategy] = {
            "config": config.STRATEGIES[strategy],
            "chunks": len(chunks), "count_before": before, "count_after": after,
            "embedded_this_run": len(todo), "deleted_stale": len(stale),
            "duplicates_in_store": after - len(new_ids),
            "chunk_chars": length_stats([len(c.text) for c in chunks]),
            "chunk_tokens": length_stats(tokens),
            "chunks_over_model_max_tokens": sum(t > embedder.max_seq_length for t in tokens),
            "seconds": {"chunking": round(t_chunk, 2), "embedding": round(t_embed, 2), "store": round(t_store, 2)},
            "cost_usd": 0.0,
        }
        print(strategy, json.dumps(report["strategies"][strategy]["seconds"]),
              f"chunks={len(chunks)} embedded={len(todo)} before={before} after={after}")
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    with (config.RESULTS / "ingest_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report


def query(store: ChromaStore, embedder: E5Embedder, split: str) -> list[dict]:
    bench = json.loads(config.BENCHMARK.read_text(encoding="utf-8"))
    qs = [q for q in bench["questions"] if split == "all" or q["split"] == split]
    traces = []
    t0 = time.perf_counter()
    for q in qs:
        qvec = embedder.embed_query(q["question"])  # same query vector for all strategies
        for strategy, cfg in config.STRATEGIES.items():
            applied = q.get("filter") if cfg["use_filter"] else None   # C: filter only if the question has one
            hits = store.search(strategy, qvec, top_k=config.TOP_K, filters=applied)
            traces.append(evaluate_query(q, hits, strategy, applied))
    elapsed = time.perf_counter() - t0
    out_dir = config.RESULTS
    with (out_dir / f"traces_{split}.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    summary = {"run_at": _now(), "split": split, "model": embedder.model_name, "top_k": config.TOP_K,
               "query_seconds_total": round(elapsed, 2), "rows": summarize(traces)}
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in summary["rows"]:
        print(f"{r['strategy']} {r['split']}: Hit@1 {r['hit1']}/{r['N']}  Evidence Hit@3 {r['hit3']}/{r['N']}  "
              f"gold page in top3 {r['gold_page_top3']}/{r['N']}  filter compliant "
              f"{r['filter_compliant']}/{r['filter_questions']}")
    return traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["ingest", "query"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = ap.parse_args()
    embedder = E5Embedder()
    store = ChromaStore(config.CHROMA_DIR, embedder.dimensions)
    if args.command == "ingest":
        ingest(store, embedder)
    else:
        for s in config.STRATEGIES:
            if store.count(s) == 0:
                raise SystemExit(f"collection for strategy {s} is empty; run ingest first")
        query(store, embedder, args.split)


if __name__ == "__main__":
    main()
