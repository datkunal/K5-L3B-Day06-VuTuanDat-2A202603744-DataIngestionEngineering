"""Step 1: read + normalize the corpus, write pages.jsonl, manifest and chunk statistics.

Run from starter/:  python -m src.prepare
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from datetime import datetime, timezone

import pypdfium2

from . import config
from .chunking import by_paragraph, fixed_size
from .ingestion import load_pdf, read_outline, section_lookup
from .models import DocumentPage


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pages() -> list[DocumentPage]:
    rows = [json.loads(l) for l in config.PAGES_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [DocumentPage(**r) for r in rows]


def build_chunks(pages: list[DocumentPage], strategy: str):
    outline = read_outline(config.RAW_PDF)
    heading_for = section_lookup(outline)
    cfg = config.STRATEGIES[strategy]
    out = []
    for p in pages:
        if cfg["chunking"] == "fixed_size":
            out += fixed_size(p, cfg["size"], cfg["overlap"], strategy=strategy, heading_for=heading_for)
        else:
            out += by_paragraph(p, cfg["max_size"], strategy=strategy, heading_for=heading_for)
    return out


def length_stats(lengths: list[int]) -> dict:
    s = sorted(lengths)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return {"count": len(s), "total_chars": sum(s), "mean": round(statistics.mean(s), 1),
            "median": statistics.median(s), "min": s[0], "p10": q(0.10), "p90": q(0.90), "max": s[-1]}


def main() -> None:
    if not config.RAW_PDF.exists():
        raise SystemExit(f"Missing {config.RAW_PDF}. Run: python scripts/download_corpus.py")
    outline = read_outline(config.RAW_PDF)
    pages, errors = load_pdf(config.RAW_PDF, config.DOCUMENT_ID, config.SOURCE, config.CONTENT_PAGES, outline)
    if len(pages) < 30:
        raise SystemExit(f"Only {len(pages)} content pages; need >= 30")

    config.PAGES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with config.PAGES_JSONL.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    n_physical = len(pypdfium2.PdfDocument(str(config.RAW_PDF)))
    manifest = {
        "sources": [{
            "source_id": config.DOCUMENT_ID,
            "title": "Introduction to Intellectual Property (OpenStax, 2021)",
            "authors": "Senior contributing authors David Kline, David Kappos; The Michelson 20MM Foundation",
            "source_path_or_url": config.SOURCE_URL,
            "local_path": config.SOURCE,
            "license_or_permission": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), stated on PDF page 4. "
                                     "OpenStax/Rice University names and logos are excluded from the license.",
            "version_or_snapshot": "WEB PDF; original publication year 2021 (PDF page 4); "
                                   f"sha256={sha256(config.RAW_PDF)}",
            "accessed_at": datetime.now(timezone.utc).date().isoformat(),
            "physical_pages": n_physical,
            "page_counting_rule": (
                "page = 1-based physical PDF page index. Counted: body pages of chapters 1-5 "
                f"{config.CHAPTER_PAGE_RANGES}. Excluded: cover/front matter/preface (1-16), chapter-opener "
                "pages that only contain a figure and credit (17, 65, 101, 171), assessment-question pages, "
                "glossary, answer key and index (188-201)."),
            "counted_pages": [p.page for p in pages],
            "counted_page_count": len(pages),
            "chars_per_page": {p.page: p.metadata["char_count"] for p in pages},
            "total_chars": sum(p.metadata["char_count"] for p in pages),
            "changes": "Text extracted with pypdfium2; Unicode NFKC, running footers removed, wrapped lines "
                       "rejoined into paragraphs. No content rewritten.",
        }],
        "ingestion_errors": errors,
    }
    config.MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    # Page-level extraction problems (empty/short pages); pipeline/store errors: results/ingestion_errors.json
    (config.RESULTS / "pdf_page_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    stats = {}
    for s in ("A", "B"):
        chunks = build_chunks(pages, s)
        stats[s] = length_stats([len(c.text) for c in chunks])
    print(json.dumps({"pages": len(pages), "errors": errors, "chunk_stats": stats}, indent=2))


if __name__ == "__main__":
    main()
