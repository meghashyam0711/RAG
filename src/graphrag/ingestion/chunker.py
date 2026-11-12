"""Contextual, structure-aware chunking.

* Text blocks are packed into ~``chunk_tokens`` chunks with ~10% overlap and
  never cross a section boundary.
* Tables are emitted as *atomic* chunks. A table larger than the budget is
  split by rows, and every split repeats the header row so no row is ever
  separated from its column names.
* Every chunk gets a short metadata header (doc, domain, section, pages,
  units) that is prepended before embedding, so scale and temporal context
  ("$ in millions", "FY2022") are never severed from the numbers.
"""

from __future__ import annotations

import hashlib
import re

from ..models import Block, BlockType, Chunk

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def approx_tokens(text: str) -> int:
    return max(1, int(len(_TOKEN_RE.findall(text)) * 0.75))


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{doc_id}:{idx}:{text[:200]}".encode()).hexdigest()[:10]
    return f"{doc_id}::c{idx:04d}-{h}"


def make_header(doc_name: str, domain: str, section: str, pages: tuple[int | None, int | None], units: str | None,
                extra: dict[str, str] | None = None) -> str:
    parts = [f"Doc: {doc_name}", f"Domain: {domain}"]
    if section:
        parts.append(f"Section: {section}")
    if pages[0] is not None:
        parts.append(f"Pages: {pages[0]}" + (f"-{pages[1]}" if pages[1] and pages[1] != pages[0] else ""))
    if units:
        parts.append(f"Units: {units}")
    for k, v in (extra or {}).items():
        parts.append(f"{k}: {v}")
    return "[" + " | ".join(parts) + "]"


def chunk_blocks(
    blocks: list[Block],
    *,
    doc_id: str,
    doc_name: str,
    domain: str,
    chunk_tokens: int = 800,
    overlap: float = 0.10,
    extra_meta: dict[str, str] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[Block] = []
    buf_tokens = 0
    overlap_tokens = int(chunk_tokens * overlap)

    def emit(text: str, ctype: str, section: str, p0, p1, units=None) -> None:
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc_id, idx, text),
                doc_id=doc_id,
                doc_name=doc_name,
                domain=domain,
                text=text,
                header=make_header(doc_name, domain, section, (p0, p1), units, extra_meta),
                chunk_type=ctype,  # type: ignore[arg-type]
                section=section,
                page_start=p0,
                page_end=p1,
                units=units,
            )
        )

    def flush_text(keep_overlap: bool) -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf)
        pages = [b.page for b in buf if b.page is not None]
        emit(text, "text", buf[0].section, min(pages) if pages else None, max(pages) if pages else None)
        if keep_overlap and overlap_tokens > 0:
            tail = _tail_tokens(text, overlap_tokens)
            last = buf[-1]
            buf = [Block(type=BlockType.text, text=tail, page=last.page, section=last.section)]
            buf_tokens = approx_tokens(tail)
        else:
            buf, buf_tokens = [], 0

    for block in blocks:
        if block.type == BlockType.heading:
            flush_text(keep_overlap=False)  # never straddle sections
            # Keep the heading text in the next chunk so it is searchable.
            buf = [block]
            buf_tokens = approx_tokens(block.text)
            continue

        if block.type == BlockType.table:
            # A preceding heading-only buffer is context for the table, not a chunk of its own.
            lead = ""
            if buf and all(b.type == BlockType.heading for b in buf):
                lead = buf[0].text
                buf, buf_tokens = [], 0
            else:
                flush_text(keep_overlap=False)
            for part in _split_table(block.text, chunk_tokens):
                text = f"{lead}\n\n{part}".strip() if lead else part
                emit(text, "table", block.section, block.page, block.page, block.units)
            continue

        t = approx_tokens(block.text)
        if t > chunk_tokens:  # very long paragraph: split by sentences
            for piece in _split_long(block.text, chunk_tokens):
                if buf_tokens + approx_tokens(piece) > chunk_tokens:
                    flush_text(keep_overlap=True)
                buf.append(Block(type=BlockType.text, text=piece, page=block.page, section=block.section))
                buf_tokens += approx_tokens(piece)
            continue
        if buf_tokens + t > chunk_tokens:
            flush_text(keep_overlap=True)
        buf.append(block)
        buf_tokens += t

    if buf and not all(b.type == BlockType.heading for b in buf):
        flush_text(keep_overlap=False)
    return chunks


def _tail_tokens(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[-max(1, int(n / 0.75)) :])


def _split_long(text: str, budget: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sentences:
        if cur and approx_tokens(cur + " " + s) > budget:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _split_table(md: str, budget: int) -> list[str]:
    if approx_tokens(md) <= budget:
        return [md]
    lines = md.splitlines()
    # header = caption lines + header row + separator row
    sep_idx = next((i for i, l in enumerate(lines) if re.match(r"^\s*\|?\s*:?-{2,}", l)), 1)
    header = lines[: sep_idx + 1]
    rows = lines[sep_idx + 1 :]
    parts, cur = [], list(header)
    for row in rows:
        if approx_tokens("\n".join(cur + [row])) > budget and len(cur) > len(header):
            parts.append("\n".join(cur))
            cur = list(header)
        cur.append(row)
    if len(cur) > len(header):
        parts.append("\n".join(cur))
    return parts
