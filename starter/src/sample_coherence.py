"""Fix the 10 source positions for the chunk-coherence audit (rule set BEFORE seeing scores).

Rule: take every 15th content page (sorted page ids, indices 0, 15, ..., 135 -> 10 pages); on each
page the position is the midpoint of the first paragraph with >= 300 characters. For each strategy
the audited chunk is (fallback if no paragraph reaches 300 chars: the longest paragraph) the first chunk of that page whose [char_start, char_end) covers the position.

Writes results/coherence_sample.csv with empty label columns to fill manually.
Run from starter/:  python -m src.sample_coherence
"""
from __future__ import annotations

import csv

from . import config
from .prepare import build_chunks, load_pages


def main() -> None:
    pages = sorted(load_pages(), key=lambda p: p.page)
    sample_pages = [pages[i] for i in range(0, len(pages), 15)][:10]
    positions = []
    for p in sample_pages:
        paras, off = [], 0
        for para in p.text.split("\n\n"):
            paras.append((off, para))
            off += len(para) + 2
        # first paragraph >= 300 chars; fallback: the longest paragraph on the page
        start, para = next(((o, t) for o, t in paras if len(t) >= 300), max(paras, key=lambda x: len(x[1])))
        positions.append((p.page, start + len(para) // 2, para[:80]))
    chunks = {s: build_chunks(sample_pages, s) for s in ("A", "B")}  # C uses B's chunks
    rows = []
    for page, pos, para_start in positions:
        for s in ("A", "B", "C"):
            src = "B" if s == "C" else s
            c = next(c for c in chunks[src] if c.page == page
                     and c.metadata["char_start"] <= pos < c.metadata["char_end"])
            rows.append({"position_id": f"p{page:03d}@{pos}", "page": page, "char_offset": pos,
                         "paragraph_start": para_start, "strategy": s,
                         "chunk_id": c.chunk_id.replace("B-", "C-", 1) if s == "C" else c.chunk_id,
                         "chunk_chars": len(c.text), "chunk_text": c.text,
                         "coherence_0_2": "", "reason": ""})
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS / "coherence_sample.csv"
    # Never wipe manual labels: carry them over when the sampled chunk is unchanged.
    if out.exists():
        with out.open(encoding="utf-8", newline="") as f:
            old = {(r["position_id"], r["strategy"]): r for r in csv.DictReader(f)}
        for r in rows:
            prev = old.get((r["position_id"], r["strategy"]))
            if prev and prev["chunk_text"] == r["chunk_text"]:
                r["coherence_0_2"], r["reason"] = prev["coherence_0_2"], prev["reason"]
            elif prev and prev["coherence_0_2"]:
                print(f"WARNING: chunk changed for {r['position_id']} {r['strategy']}; label must be redone")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(positions)} positions x 3 strategies -> results/coherence_sample.csv")


if __name__ == "__main__":
    main()
