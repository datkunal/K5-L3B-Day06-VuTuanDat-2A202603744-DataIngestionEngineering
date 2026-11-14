"""Automatic retrieval metrics (Hit@1, Evidence Hit@3) and per-query traces.

A retrieved chunk is a *hit* when it comes from the gold source and one of the gold pages AND
contains the full evidence quote (after light normalization). Coherence and grounding labels
are NOT inferred here; they are assigned manually in results/manual_labels.csv.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def norm(text: str) -> str:
    """Normalization used only for evidence matching (case, quotes, dashes, whitespace)."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("—", "-").replace("–", "-")
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def is_hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    md = hit["metadata"]
    return (md.get("document_id") == question["gold_source_id"]
            and int(md.get("page")) in question["gold_page"]
            and norm(question["evidence_quote"]) in norm(hit["text"]))


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]], strategy: str,
                   applied_filter: dict | None) -> dict[str, Any]:
    flags = [is_hit(question, h) for h in hits]
    gold_page_retrieved = [int(h["metadata"]["page"]) in question["gold_page"] for h in hits]
    filter_ok = None
    if applied_filter:
        filter_ok = all(all(h["metadata"].get(k) == v for k, v in applied_filter.items()) for h in hits)
    return {
        "strategy": strategy,
        "id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "question_filter": question.get("filter"),
        "applied_filter": applied_filter,
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "hit_at_1": bool(flags[:1] and flags[0]),
        "hit_at_3": any(flags[:3]),
        "first_hit_rank": next((i + 1 for i, f in enumerate(flags) if f), None),
        "gold_page_in_top3": any(gold_page_retrieved[:3]),
        "filter_compliant": filter_ok,
        "top_k": [{**h, "is_evidence_hit": f} for h, f in zip(hits, flags)],
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for strategy in sorted({t["strategy"] for t in traces}):
        for split in ("dev", "test"):
            ts = [t for t in traces if t["strategy"] == strategy and t["split"] == split]
            if not ts:
                continue
            n = len(ts)
            filt = [t for t in ts if t["applied_filter"]]
            rows.append({
                "strategy": strategy, "split": split, "N": n,
                "hit1": sum(t["hit_at_1"] for t in ts),
                "hit3": sum(t["hit_at_3"] for t in ts),
                "gold_page_top3": sum(t["gold_page_in_top3"] for t in ts),
                "filter_questions": len(filt),
                "filter_compliant": sum(bool(t["filter_compliant"]) for t in filt),
            })
    return rows
