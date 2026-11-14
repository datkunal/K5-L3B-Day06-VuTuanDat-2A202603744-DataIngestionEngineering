"""Read and normalize the PDF corpus page by page, keeping traceable metadata."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from pypdf import PdfReader

from .models import DocumentPage

# Chapter number -> intellectual-property type (taken from the book's chapter titles).
IP_TYPE_BY_CHAPTER = {1: "patent", 2: "patent", 3: "copyright", 4: "trademark", 5: "trade_secret"}

# Running footers printed by OpenStax, e.g. "10 1 • Patent Basics" (even pages)
# and "1.7 • Criteria for Patenting 43" (odd pages), plus the access notice.
_FOOTER_EVEN_TAIL = re.compile(r"(?:^|(?<=[\s\-]))(?P<num>\d{1,3})\s+\d\s*•\s*[A-Z][^•]*$")
_FOOTER_ODD_TAIL = re.compile(r"(?:^|(?<=\s))\d(?:\.\d{1,2})?\s*•\s*[A-Z][^•]*?\s(?P<num>\d{1,3})$")
_ACCESS = re.compile(r"^Access for free at openstax\.org$")
_SECTION_RE = re.compile(r"^(\d)\.(\d{1,2})\s+\S")
# A visual line shorter than this is treated as the end of a paragraph/heading.
_SHORT_LINE = 80


def read_outline(path: Path) -> list[dict[str, Any]]:
    """Return outline entries (title, 1-based start page, depth) from the PDF bookmarks."""
    reader = PdfReader(str(path))
    out: list[dict[str, Any]] = []

    def walk(items, depth=0):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                title = unicodedata.normalize("NFKC", item.title).replace("�", "'")
                out.append({"title": title.strip(), "page": reader.get_destination_page_number(item) + 1,
                            "depth": depth})

    walk(reader.outline)
    return out


def normalize_text(text: str) -> tuple[str, int | None]:
    """Normalize Unicode/whitespace, drop running footers and rebuild paragraphs.

    Returns (normalized_text, printed_page_number_or_None). Paragraphs are separated by a
    blank line; wrapped visual lines inside a paragraph are joined with a space.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x02", "-")  # pdfium marks a hyphen at a line break with \x02
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not _ACCESS.match(ln)]

    # The running footer is the last line, sometimes glued to the end of the last text line.
    printed_page = None
    if lines:
        last = lines[-1]
        m = _FOOTER_EVEN_TAIL.search(last) or _FOOTER_ODD_TAIL.search(last)
        if m:
            printed_page = int(m.group("num"))
            lines[-1] = last[: m.start()].rstrip()
            if not lines[-1]:
                lines.pop()
        elif re.fullmatch(r"\d{1,3}", last):  # chapter-opener pages: bare chapter number
            lines.pop()
    kept = lines

    paragraphs: list[str] = []
    current: list[str] = []
    for i, ln in enumerate(kept):
        current.append(ln)
        nxt = kept[i + 1] if i + 1 < len(kept) else ""
        # A short visual line ends a paragraph/heading/list item, unless the sentence clearly
        # continues (line ends with , ( / - or the next line starts in lower case).
        continues = ln.endswith((",", "(", "/", "-", "–", "—")) or (nxt[:1].islower())
        if len(ln) < _SHORT_LINE and not continues:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(p for p in paragraphs if p), printed_page


def _section_index(outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Numbered sections (e.g. '1.7 Criteria for Patenting') sorted by start page."""
    sections = []
    for e in outline:
        m = _SECTION_RE.match(e["title"])
        if m:
            sections.append({"section": f"{m.group(1)}.{m.group(2)}", "chapter": int(m.group(1)),
                             "title": e["title"], "page": e["page"]})
    return sorted(sections, key=lambda s: (s["page"], s["chapter"], int(s["section"].split(".")[1])))


def load_pdf(path: Path, document_id: str, source: str, content_pages: list[int],
             outline: list[dict[str, Any]] | None = None) -> tuple[list[DocumentPage], list[dict]]:
    """Extract the selected physical pages; return (pages, error_log).

    `content_pages` are 1-based physical page numbers counted as content in the manifest.
    Each page records chapter, ip_type and the section active at the *start* of the page;
    sections that begin on the page are listed in `sections_on_page`.
    """
    outline = outline if outline is not None else read_outline(path)
    sections = _section_index(outline)
    # Top-level entries only: the glossary/answer key also contain "Chapter N" sub-entries.
    chapters = {int(m.group(1)): e["page"] for e in outline
                if e["depth"] == 0 and (m := re.match(r"^Chapter (\d+)\b", e["title"]))}
    pdf = pdfium.PdfDocument(str(path))
    pages: list[DocumentPage] = []
    errors: list[dict] = []
    for pno in content_pages:
        if pno < 1 or pno > len(pdf):
            errors.append({"page": pno, "error": "page out of range"})
            continue
        raw = pdf[pno - 1].get_textpage().get_text_bounded()
        text, printed = normalize_text(raw)
        if len(text) < 200:
            errors.append({"page": pno, "error": f"too little text ({len(text)} chars); possibly image-only"})
            if not text:
                continue
        chapter = max((c for c, start in chapters.items() if start <= pno), default=None)
        started_before = [s for s in sections if s["page"] < pno and s["chapter"] == chapter]
        starting_here = [s for s in sections if s["page"] == pno]
        # Section active at the top of the page: the last section of this chapter that started
        # on an earlier page; on a chapter's first section page, that first section.
        active = started_before[-1] if started_before else (starting_here[0] if starting_here else None)
        pages.append(DocumentPage(
            document_id=document_id, source=source, page=pno, text=text,
            metadata={
                "printed_page": printed,
                "chapter": chapter,
                "ip_type": IP_TYPE_BY_CHAPTER.get(chapter),
                "section": active["section"] if active else None,
                "section_title": active["title"] if active else None,
                "sections_on_page": [s["section"] for s in starting_here],
                "char_count": len(text),
            }))
    return pages, errors


def section_lookup(outline: list[dict[str, Any]]):
    """Return a function mapping (page, paragraph text) -> section dict, used by chunking
    to switch section when a section heading appears in the middle of a page."""
    sections = _section_index(outline)
    by_page: dict[int, list[dict]] = {}
    for s in sections:
        by_page.setdefault(s["page"], []).append(s)

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'")).strip().lower()

    def heading_for(page: int, paragraph: str) -> dict | None:
        p = _norm(paragraph)
        for s in by_page.get(page, []):
            # e.g. heading line "1.7 Criteria for Patenting"
            if p.startswith(_norm(s["title"])):
                return s
        return None

    return heading_for
