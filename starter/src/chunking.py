"""Fixed-size (strategy A) and paragraph/structure-aware (strategy B/C) chunking.

Chunks never cross a page boundary, so every chunk keeps exactly one source page.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable

from .models import Chunk, DocumentPage

HeadingFn = Callable[[int, str], "dict | None"]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _section_marks(page: DocumentPage, heading_for: HeadingFn | None) -> list[tuple[int, dict]]:
    """(char offset, section info) where a section is active, starting with the page's own."""
    md = page.metadata
    marks = [(0, {"section": md.get("section"), "title": md.get("section_title"),
                  "chapter": md.get("chapter")})]
    if heading_for is None:
        return marks
    offset = 0
    for para in page.text.split("\n\n"):
        s = heading_for(page.page, para)
        if s:
            marks.append((offset, s))
        offset += len(para) + 2
    return marks


def _section_at(marks: list[tuple[int, dict]], offset: int) -> dict:
    current = marks[0][1]
    for pos, s in marks:
        if pos <= offset:
            current = s
    return current


def _make_chunk(page: DocumentPage, strategy: str, idx: int, text: str, start: int,
                marks: list[tuple[int, dict]]) -> Chunk:
    sec = _section_at(marks, start)
    md = page.metadata
    metadata = {
        "strategy": strategy,
        "printed_page": md.get("printed_page") if md.get("printed_page") is not None else -1,
        "chapter": md.get("chapter"),
        "ip_type": md.get("ip_type"),
        "section": sec.get("section") or "",
        "section_title": sec.get("title") or "",
        "char_start": start,
        "char_end": start + len(text),
        "content_sha1": _sha1(text),
    }
    return Chunk(chunk_id=f"{strategy}-{page.document_id}-p{page.page:03d}-c{idx:02d}",
                 document_id=page.document_id, source=page.source, page=page.page,
                 text=text, metadata=metadata)


def fixed_size(page: DocumentPage, size: int, overlap: int, strategy: str = "A",
               heading_for: HeadingFn | None = None) -> list[Chunk]:
    """Character windows of `size` with `overlap`, ignoring sentence/paragraph structure."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text, marks = page.text, _section_marks(page, heading_for)
    chunks, start, idx = [], 0, 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(_make_chunk(page, strategy, idx, piece, start, marks))
            idx += 1
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?”\"])\s+(?=[A-Z“\"(])")


def _split_long(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries (hard-cut only a giant sentence)."""
    out, buf = [], ""
    for sent in _SENT_SPLIT.split(paragraph):
        while len(sent) > max_size:  # pathological sentence: hard cut
            if buf:
                out.append(buf)
                buf = ""
            out.append(sent[:max_size])
            sent = sent[max_size:]
        if buf and len(buf) + 1 + len(sent) > max_size:
            out.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        out.append(buf)
    return out


def _is_heading(para: str) -> bool:
    return len(para) < 80 and not para.rstrip().endswith((".", ":", ";", ",", "?", "!", "”", '"'))


def by_paragraph(page: DocumentPage, max_size: int, strategy: str = "B",
                 heading_for: HeadingFn | None = None) -> list[Chunk]:
    """Greedily pack whole paragraphs up to `max_size` characters.

    - A numbered section heading always starts a new chunk (no chunk mixes two sections).
    - A trailing short heading is carried over to the next chunk, so it stays with its body.
    - Paragraphs longer than `max_size` are split at sentence boundaries.
    """
    marks = _section_marks(page, heading_for)
    section_starts = {pos for pos, _ in marks[1:]}

    units: list[tuple[int, str]] = []  # (offset in page text, text)
    offset = 0
    for para in page.text.split("\n\n"):
        if len(para) > max_size:
            sub_off = offset
            for piece in _split_long(para, max_size):
                found = page.text.find(piece[:40], sub_off)
                sub_off = found if found >= 0 else sub_off
                units.append((sub_off, piece))
                sub_off += len(piece)
        elif para.strip():
            units.append((offset, para))
        offset += len(para) + 2

    groups: list[list[tuple[int, str]]] = []
    buf: list[tuple[int, str]] = []
    for off, txt in units:
        size_if_added = sum(len(t) + 2 for _, t in buf) + len(txt)
        if buf and (off in section_starts or size_if_added > max_size):
            carry: list[tuple[int, str]] = []
            # Carry trailing short headings into the next chunk if it still fits in max_size.
            while (buf and _is_heading(buf[-1][1])
                   and sum(len(t) + 2 for _, t in carry) + len(buf[-1][1]) + 2 + len(txt) <= max_size):
                carry.insert(0, buf.pop())
            if buf:
                groups.append(buf)
            buf = carry
        buf.append((off, txt))
    if buf:
        groups.append(buf)

    chunks = []
    for idx, group in enumerate(groups):
        text = "\n\n".join(t for _, t in group)
        chunks.append(_make_chunk(page, strategy, idx, text, group[0][0], marks))
    return chunks
