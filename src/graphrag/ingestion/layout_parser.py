"""Layout-aware parsing into reading-ordered Blocks.

Backends, best first:
  1. Docling (``pip install '.[docling]'``): bounding-box reading order for
     multi-column pages, tables exported as intact Markdown, page provenance.
  2. Markdown / text / HTML: parsed natively (tables kept atomic).
  3. pypdf: last-resort PDF text extraction (no table structure). A warning is
     emitted because multi-column specs and tables degrade in this mode.

Regardless of backend, the parser:
  * keeps tables atomic (never split across chunks),
  * preserves parenthetical negatives verbatim, e.g. ``(1,234)``,
  * attaches the nearest unit/scale header ("in millions") to each table,
  * tracks the current section heading for every block.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

from ..models import Block, BlockType

log = logging.getLogger(__name__)

SCALE_RE = re.compile(r"(?:\$\s*)?in\s+(thousands|millions|billions)(?:,?\s*except[^)\n]*)?", re.I)
HEADING_MD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PAGE_MARK_RE = re.compile(r"<!--\s*page[:\s]+(\d+)\s*-->", re.I)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def parse_document(path: str | Path) -> tuple[list[Block], str]:
    """Return (blocks, backend_name)."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown(path.read_text(encoding="utf-8", errors="replace")), "markdown"
    if suffix in {".html", ".htm"}:
        return parse_markdown(_html_to_markdown(path.read_text(encoding="utf-8", errors="replace"))), "html"

    docling_blocks = _try_docling(path)
    if docling_blocks is not None:
        return docling_blocks, "docling"

    if suffix == ".pdf":
        log.warning(
            "Docling not installed: falling back to pypdf for %s. Tables and multi-column "
            "layouts will lose structure. Install with: pip install '.[docling]'",
            path.name,
        )
        return _parse_pdf_pypdf(path), "pypdf"

    raise ValueError(f"Unsupported file type {suffix!r}. Install docling for DOCX/PPTX/images.")


# ---------------------------------------------------------------------------
# Docling
# ---------------------------------------------------------------------------

def _try_docling(path: Path) -> list[Block] | None:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError:
        return None
    try:
        result = DocumentConverter().convert(str(path))
        doc = result.document
    except Exception as exc:  # pragma: no cover - depends on docling runtime
        log.warning("Docling failed on %s (%s); falling back", path.name, exc)
        return None

    blocks: list[Block] = []
    section = ""
    recent_text: list[str] = []
    for item, _level in doc.iterate_items():  # reading order, column-aware
        label = str(getattr(item, "label", "")).lower()
        page = None
        prov = getattr(item, "prov", None)
        if prov:
            page = getattr(prov[0], "page_no", None)
        if "table" in label:
            try:
                md = item.export_to_markdown(doc=doc)
            except TypeError:  # older docling
                md = item.export_to_markdown()
            caption = ""
            try:
                caption = item.caption_text(doc) or ""
            except Exception:
                pass
            units = _find_units(caption + "\n" + md + "\n" + "\n".join(recent_text[-3:]))
            text = f"{caption}\n{md}".strip() if caption else md
            blocks.append(Block(type=BlockType.table, text=text, page=page, section=section, units=units))
            continue
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if "header" in label or "title" in label:
            section = text
            blocks.append(Block(type=BlockType.heading, text=text, page=page, section=section))
        else:
            blocks.append(Block(type=BlockType.text, text=text, page=page, section=section))
        recent_text.append(text)
    return blocks


# ---------------------------------------------------------------------------
# Markdown (also the canonical intermediate format)
# ---------------------------------------------------------------------------

def parse_markdown(md: str) -> list[Block]:
    blocks: list[Block] = []
    section = ""
    page: int | None = None
    para: list[str] = []
    lines = md.splitlines()
    i = 0
    recent_text: list[str] = []

    def flush_para() -> None:
        if para:
            text = "\n".join(para).strip()
            if text:
                blocks.append(Block(type=BlockType.text, text=text, page=page, section=section))
                recent_text.append(text)
            para.clear()

    while i < len(lines):
        line = lines[i]
        pm = PAGE_MARK_RE.search(line)
        if pm:
            flush_para()
            page = int(pm.group(1))
            i += 1
            continue
        hm = HEADING_MD_RE.match(line)
        if hm:
            flush_para()
            section = hm.group(2).strip()
            blocks.append(Block(type=BlockType.heading, text=section, page=page, section=section))
            i += 1
            continue
        if TABLE_ROW_RE.match(line):
            flush_para()
            rows = []
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            # Caption = the line right after the table if it looks like one, else preceding text.
            context = "\n".join(recent_text[-2:])
            units = _find_units("\n".join(rows[:2]) + "\n" + context)
            blocks.append(Block(type=BlockType.table, text="\n".join(rows), page=page, section=section, units=units))
            continue
        if not line.strip():
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()
    return blocks


def _find_units(text: str) -> str | None:
    m = SCALE_RE.search(text)
    if m:
        scale = m.group(1).lower()
        currency = "$ " if "$" in text[max(0, m.start() - 3) : m.end()] or "$" in text else ""
        return f"{currency}in {scale}"
    m = re.search(r"\(\s*(ms|milliseconds|seconds|requests\s*/\s*\w+|MB|GB|MiB|GiB)\s*\)", text, re.I)
    if m:
        return m.group(1)
    return None


def _html_to_markdown(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)

    def table_to_md(m: re.Match) -> str:
        rows = re.findall(r"(?is)<tr.*?>(.*?)</tr>", m.group(0))
        out = []
        for r_i, row in enumerate(rows):
            cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"(?is)<t[hd].*?>(.*?)</t[hd]>", row)]
            out.append("| " + " | ".join(cells) + " |")
            if r_i == 0:
                out.append("|" + "---|" * len(cells))
        return "\n" + "\n".join(out) + "\n"

    raw = re.sub(r"(?is)<table.*?</table>", table_to_md, raw)
    for n in range(1, 7):
        raw = re.sub(rf"(?is)<h{n}[^>]*>(.*?)</h{n}>", lambda m, n=n: "\n" + "#" * n + " " + m.group(1) + "\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(raw)


# ---------------------------------------------------------------------------
# pypdf fallback
# ---------------------------------------------------------------------------

def _parse_pdf_pypdf(path: Path) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    md_parts = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        md_parts.append(f"<!-- page: {page_no} -->\n")
        for line in text.splitlines():
            stripped = line.strip()
            # Heuristic headings: short numbered or ALL CAPS lines.
            if re.match(r"^(\d+(\.\d+)*\.?|Item\s+\d+[A-Z]?\.?|Appendix\s+[A-Z]|Note\s+\d+)\s+\S", stripped) and len(stripped) < 90:
                md_parts.append(f"\n## {stripped}\n")
            elif stripped.isupper() and 3 < len(stripped) < 80:
                md_parts.append(f"\n## {stripped.title()}\n")
            else:
                md_parts.append(line)
        md_parts.append("\n")
    return parse_markdown("\n".join(md_parts))


def blocks_to_markdown(blocks: list[Block]) -> str:
    out: list[str] = []
    last_page = None
    for b in blocks:
        if b.page is not None and b.page != last_page:
            out.append(f"<!-- page: {b.page} -->")
            last_page = b.page
        if b.type == BlockType.heading:
            out.append(f"## {b.text}")
        else:
            out.append(b.text)
        out.append("")
    return "\n".join(out)
