"""Check every benchmark question before any retrieval run.

- required fields, unique ids, >=15 dev / >=5 test, >=3 filter questions with >=1 in test
- evidence_quote occurs on the gold page of the normalized corpus
- filter values exist in the page metadata (not invented) and match the gold page
- no test question shares a gold page with a dev question

Run from starter/:  python -m src.verify_benchmark
"""
from __future__ import annotations

import json
import sys

from . import config
from .evaluation import norm
from .prepare import load_pages

REQUIRED = ["id", "split", "question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote"]


def main() -> int:
    bench = json.loads(config.BENCHMARK.read_text(encoding="utf-8"))
    qs = bench["questions"]
    pages = {p.page: p for p in load_pages()}
    problems, report = [], []

    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    filt = [q for q in qs if q.get("filter")]
    if len(qs) < 20 or len(dev) < 15 or len(test) < 5:
        problems.append(f"need >=20 questions, >=15 dev, >=5 test (got {len(qs)}, {len(dev)}, {len(test)})")
    if len(filt) < 3 or not any(q["split"] == "test" for q in filt):
        problems.append("need >=3 filter questions with >=1 in test")

    for q in qs:
        missing = [k for k in REQUIRED if q.get(k) in (None, "", [])]
        if missing:
            problems.append(f"{q['id']}: missing {missing}")
            continue
        found_on = [p for p in q["gold_page"] if p in pages and norm(q["evidence_quote"]) in norm(pages[p].text)]
        # Where else does the quote occur? (would be additional valid evidence pages)
        elsewhere = [n for n, p in pages.items() if n not in q["gold_page"] and norm(q["evidence_quote"]) in norm(p.text)]
        if not found_on:
            problems.append(f"{q['id']}: evidence not found on gold page(s) {q['gold_page']}")
        if elsewhere:
            problems.append(f"{q['id']}: evidence also occurs on pages {elsewhere}; add them as gold pages")
        if q.get("filter"):
            for k, v in q["filter"].items():
                values = {p.metadata.get(k) for p in pages.values()}
                if v not in values:
                    problems.append(f"{q['id']}: filter value {k}={v} does not exist in metadata")
                if any(pages[g].metadata.get(k) != v for g in q["gold_page"] if g in pages):
                    problems.append(f"{q['id']}: gold page metadata does not match filter {k}={v}")
        report.append({"id": q["id"], "split": q["split"], "gold_page": q["gold_page"],
                       "evidence_found_on": found_on, "filter": q.get("filter")})

    dev_pages = {p for q in dev for p in q["gold_page"]}
    overlap = [q["id"] for q in test if set(q["gold_page"]) & dev_pages]
    if overlap:
        problems.append(f"test questions share gold pages with dev: {overlap}")

    out = {"n_questions": len(qs), "n_dev": len(dev), "n_test": len(test), "n_filter": len(filt),
           "n_filter_test": sum(q["split"] == "test" for q in filt), "problems": problems, "checks": report}
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "benchmark_verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "checks"}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
