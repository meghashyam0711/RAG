"""Schema-constrained entity/relation extraction.

Two extractors are combined for every chunk:

* **Structural (always on, deterministic)** - section membership, explicit
  cross-references ("see Appendix B", "as defined in Section 2.1"),
  override/amendment language ("supersedes the limits in Section 2"),
  code identifiers, endpoints, fiscal periods, financial metrics. This is the
  backbone that makes multi-hop retrieval work even with no LLM.
* **LLM (optional)** - Pydantic-validated extraction restricted to the
  domain's ontology (``models.ENTITY_TYPES`` / ``RELATION_TYPES``). Types
  outside the ontology are dropped, never invented.
"""

from __future__ import annotations

import logging
import re

from ..llm import LLM
from ..models import ENTITY_TYPES, RELATION_TYPES, Chunk, ExtractedEntity, ExtractedRelation, ExtractionResult

log = logging.getLogger(__name__)

_REF_KIND = r"(Section|Sec\.|§|Appendix|Annex|Note|Item|Table|Clause|Article|Exhibit|Schedule|Chapter|Part)"
_REF_ID = r"([0-9]+(?:\.[0-9]+)*[A-Za-z]?|[A-Z](?:\.[0-9]+)*|[IVXLC]+)\b"
_MODIFY_VERBS = r"(overrid(?:e|es|den|ing)|supersed(?:e|es|ed|ing)|replac(?:e|es|ed|ing)|modif(?:y|ies|ied)|amend(?:s|ed)?|chang(?:e|es|ed)|takes? precedence over)"

XREF_RE = re.compile(rf"\b{_REF_KIND}\s*{_REF_ID}", re.I)
MODIFY_FORWARD_RE = re.compile(rf"\b{_MODIFY_VERBS}\b[^.]{{0,120}}?\b{_REF_KIND}\s*{_REF_ID}", re.I)
MODIFY_PASSIVE_RE = re.compile(rf"\b(?:overridden|superseded|replaced|modified|amended|changed)\s+by\s+{_REF_KIND}\s*{_REF_ID}", re.I)

BACKTICK_RE = re.compile(r"`([^`\n]{2,60})`")
IDENT_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[a-z][a-zA-Z0-9]*(?:\.[a-z][a-zA-Z0-9_]*){1,4})\b")
ENDPOINT_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[\w/{}.:-]*)")
DEFINE_HINT_RE = re.compile(r"\b(is defined as|defaults? to|is set to|=|must be|shall be|specif(?:y|ies)|default)\b", re.I)
FISCAL_RE = re.compile(
    r"\b(?:Q[1-4]\s*(?:FY)?\s*'?\d{2,4}|[1-4]Q\s*'?\d{2,4}|FY\s*'?\d{2,4}(?:\s*Q[1-4])?|fiscal(?:\s+year)?\s+\d{4}|"
    r"(?:first|second|third|fourth)\s+(?:fiscal\s+)?quarter(?:\s+of)?\s+(?:fiscal\s+)?\d{4})\b",
    re.I,
)
ORG_RE = re.compile(r"\b([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,3}\s+(?:Inc\.?|Corp(?:oration|\.)?|LLC|Ltd\.?|plc|N\.V\.|S\.A\.|GmbH|AG))")
FIN_METRICS = [
    "net income", "net loss", "revenue", "net revenue", "total revenue", "gross margin", "gross profit", "operating income",
    "operating loss", "operating expenses", "diluted earnings per share", "earnings per share", "free cash flow",
    "cash and cash equivalents", "total assets", "total liabilities", "stockholders' equity", "long-term debt",
    "research and development", "capital expenditures", "ebitda", "inventory", "accounts receivable", "goodwill",
]
_STOP_IDENTS = {"e.g", "i.e", "etc.", "vs.", "a.m", "p.m"}


class GraphExtractor:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm

    def extract(self, chunk: Chunk) -> ExtractionResult:
        result = self._structural(chunk)
        if self.llm is not None:
            llm_result = self._llm_extract(chunk)
            if llm_result:
                result = _merge(result, llm_result)
        return result

    # ------------------------------------------------------------------
    def _structural(self, chunk: Chunk) -> ExtractionResult:
        ents: dict[tuple[str, str], ExtractedEntity] = {}
        rels: list[ExtractedRelation] = []
        text = chunk.text

        def add(name: str, type_: str, desc: str = "") -> str:
            name = name.strip()
            ents.setdefault((name.lower(), type_), ExtractedEntity(name=name, type=type_, description=desc))
            return name

        here = add(chunk.section, "Section", f"Section of {chunk.doc_name}") if chunk.section else None

        # Cross references / overrides between sections ------------------
        if here:
            modified: set[str] = set()
            for m in MODIFY_FORWARD_RE.finditer(text):
                target = add(f"{_canon_kind(m.group(2))} {m.group(3)}", "Section")
                if target.lower() != here.lower():
                    rels.append(ExtractedRelation(source=here, target=target, type="MODIFIES", evidence=_span(text, m)))
                    modified.add(target.lower())
            for m in MODIFY_PASSIVE_RE.finditer(text):
                src = add(f"{_canon_kind(m.group(1))} {m.group(2)}", "Section")
                if src.lower() != here.lower():
                    rels.append(ExtractedRelation(source=src, target=here, type="MODIFIES", evidence=_span(text, m)))
            for m in XREF_RE.finditer(text):
                kind = _canon_kind(m.group(1))
                if kind == "Item" and chunk.domain != "finance":  # "item 3" is usually prose outside 10-Ks
                    continue
                target = add(f"{kind} {m.group(2)}", "Section")
                if target.lower() != here.lower() and target.lower() not in modified:
                    rels.append(ExtractedRelation(source=here, target=target, type="REFERENCES", evidence=_span(text, m)))

        # Parameters / identifiers ---------------------------------------
        idents: set[str] = set()
        for m in BACKTICK_RE.finditer(text):
            tok = m.group(1).strip()
            if not ENDPOINT_RE.match(tok) and " " not in tok:
                idents.add(tok)
        for m in IDENT_RE.finditer(text):
            tok = m.group(1)
            if tok.lower() not in _STOP_IDENTS and not re.match(r"^\w+\.(com|org|io|md|pdf|py|json|txt)$", tok, re.I):
                idents.add(tok)
        if chunk.chunk_type == "table":
            for row in text.splitlines()[2:]:
                cells = [c.strip().strip("`") for c in row.strip().strip("|").split("|")]
                if cells and IDENT_RE.fullmatch(cells[0] or "-"):
                    idents.add(cells[0])
        for ident in idents:
            add(ident, "Parameter")
            if here:
                defining = chunk.chunk_type == "table" or bool(
                    re.search(rf"{re.escape(ident)}`?[^.\n]{{0,60}}{DEFINE_HINT_RE.pattern}", text, re.I)
                )
                rels.append(ExtractedRelation(source=here, target=ident, type="DEFINES" if defining else "MENTIONS"))

        for m in ENDPOINT_RE.finditer(text):
            ep = add(f"{m.group(1)} {m.group(2)}", "Endpoint" if chunk.domain == "technical" else "Component")
            if here:
                rels.append(ExtractedRelation(source=here, target=ep, type="DEFINES"))

        # Finance-ish -----------------------------------------------------
        periods = []
        for m in FISCAL_RE.finditer(text):
            periods.append(add(m.group(0), "FiscalPeriod"))
            if here:
                rels.append(ExtractedRelation(source=here, target=m.group(0), type="MENTIONS"))
        low = text.lower()
        for metric in FIN_METRICS:
            if re.search(rf"\b{re.escape(metric)}\b", low):
                name = add(metric.title(), "FinancialMetric")
                if here:
                    rels.append(ExtractedRelation(source=here, target=name, type="MENTIONS"))
                for p in periods[:3]:
                    rels.append(ExtractedRelation(source=name, target=p, type="REPORTED_IN"))
        for m in ORG_RE.finditer(text):
            org = add(m.group(1), "Organization")
            if here:
                rels.append(ExtractedRelation(source=here, target=org, type="MENTIONS"))

        return ExtractionResult(entities=list(ents.values()), relations=rels)

    # ------------------------------------------------------------------
    def _llm_extract(self, chunk: Chunk) -> ExtractionResult | None:
        assert self.llm is not None
        types = ENTITY_TYPES.get(chunk.domain, ENTITY_TYPES["general"])
        system = (
            "You extract a knowledge graph from a document chunk. Use ONLY these entity types: "
            f"{types}. Use ONLY these relation types: {RELATION_TYPES}. Use the exact surface form "
            "from the text for names. Prefer precise, specific entities (parameters, metrics, components, "
            "periods) over generic words. Every relation must be supported by the text; put the shortest "
            "verbatim supporting span in `evidence`. Pay attention to negation and to override language."
        )
        prompt = f"{chunk.header}\n\n{chunk.text[:6000]}"
        res = self.llm.structured(system, prompt, ExtractionResult)
        if res is None:
            return None
        allowed = set(types) | {"Section"}
        res.entities = [e for e in res.entities if e.type in allowed]
        names = {e.name.lower() for e in res.entities}
        res.relations = [
            r for r in res.relations
            if r.type in RELATION_TYPES and r.source.lower() in names and r.target.lower() in names
        ]
        return res


def _canon_kind(kind: str) -> str:
    k = kind.strip().rstrip(".").lower()
    return {"sec": "Section", "§": "Section"}.get(k, k.capitalize())


def _span(text: str, m: re.Match, pad: int = 60) -> str:
    return text[max(0, m.start() - pad) : m.end() + pad].replace("\n", " ").strip()


def _merge(a: ExtractionResult, b: ExtractionResult) -> ExtractionResult:
    seen = {(e.name.lower(), e.type) for e in a.entities}
    ents = list(a.entities) + [e for e in b.entities if (e.name.lower(), e.type) not in seen]
    rseen = {(r.source.lower(), r.target.lower(), r.type) for r in a.relations}
    rels = list(a.relations) + [r for r in b.relations if (r.source.lower(), r.target.lower(), r.type) not in rseen]
    return ExtractionResult(entities=ents, relations=rels)
