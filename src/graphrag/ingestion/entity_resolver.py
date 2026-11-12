"""Canonical entity resolution.

Prevents GraphRAG node duplication: "Q3 '22", "Third Quarter 2022" and "3Q22"
must become one node (``FiscalPeriod:FY2022_Q3``); "Appendix B" and
"APPENDIX B - Rate Limit Overrides" must become one Section node.

Strategy (cheap first):
  1. type-specific normalisers (fiscal periods, section identifiers, code identifiers)
  2. exact alias match on a normalised key
  3. RapidFuzz token-sort similarity >= threshold within the same entity type
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz, process

_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_SECTION_ID_RE = re.compile(
    r"^\s*(?:§\s*)?(section|sec\.?|appendix|annex|note|item|table|figure|chapter|clause|article|exhibit|schedule|part)\s+"
    r"([0-9]+(?:\.[0-9]+)*[a-z]?|[a-z](?:\.[0-9]+)*|[ivxlc]+)\b",
    re.I,
)
_SECTION_NUM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)\.?\s+[A-Za-z]")
_SECTION_ALIASES = {"sec": "SECTION", "sec.": "SECTION", "§": "SECTION"}


def _year(y: str) -> int:
    y = y.strip("'’ ")
    n = int(y)
    return 2000 + n if n < 100 else n


def normalize_fiscal_period(text: str) -> str | None:
    """Map many surface forms of a fiscal period to FY<yyyy>[_Q<n>|_H<n>]."""
    t = text.strip().lower().replace("’", "'")
    patterns = [
        (r"\b(?:q([1-4])|([1-4])q)\s*(?:fy|cy)?\s*'?(\d{2,4})\b", lambda m: (m.group(1) or m.group(2), m.group(3))),
        (r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:fiscal\s+)?quarter(?:\s+of)?\s+(?:fiscal\s+|fy\s*)?'?(\d{2,4})\b",
         lambda m: (str(_ORD[m.group(1)]), m.group(2))),
        (r"\bfy\s*'?(\d{2,4})\s*q([1-4])\b", lambda m: (m.group(2), m.group(1))),
    ]
    for pat, getter in patterns:
        m = re.search(pat, t)
        if m:
            q, y = getter(m)
            return f"FY{_year(y)}_Q{q}"
    m = re.search(r"\b(?:h([12])|([12])h|(first|second)\s+half(?:\s+of)?)\s*(?:fy)?\s*'?(\d{2,4})\b", t)
    if m:
        h = m.group(1) or m.group(2) or ("1" if m.group(3) == "first" else "2")
        return f"FY{_year(m.group(4))}_H{h}"
    m = re.search(r"\b(?:fy|fiscal(?:\s+year)?)\s*'?(\d{2,4})\b", t)
    if m:
        return f"FY{_year(m.group(1))}"
    return None


def section_key(text: str) -> str | None:
    """'Appendix B: Overrides' -> 'APPENDIX_B'; '2.1 Limits' -> 'SECTION_2.1'."""
    m = _SECTION_ID_RE.match(text)
    if m:
        kind = m.group(1).lower().rstrip(".")
        kind = {"sec": "section"}.get(kind, kind).upper()
        return f"{kind}_{m.group(2).upper()}"
    m = _SECTION_NUM_RE.match(text)
    if m:
        return f"SECTION_{m.group(1)}"
    return None


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w.]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", text).strip("_")[:80] or "x"


def norm_key(name: str) -> str:
    n = name.strip().strip("`'\"").lower()
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"^(the|a|an)\s+", "", n)
    return n


class EntityResolver:
    """Maps (surface name, type, doc) -> canonical node id."""

    # Section-like entities are local to a document; everything else is global
    # so the same API component / metric links across documents.
    DOC_SCOPED_TYPES = {"Section", "Clause"}

    def __init__(self, threshold: float = 92.0) -> None:
        self.threshold = threshold
        self._by_key: dict[tuple[str, str], str] = {}        # (type, normkey) -> id
        self._names_by_type: dict[str, dict[str, str]] = {}  # type -> {normkey: id}

    def register(self, entity_id: str, type_: str, names: list[str]) -> None:
        for n in names:
            k = norm_key(n)
            self._by_key[(type_, k)] = entity_id
            self._names_by_type.setdefault(type_, {})[k] = entity_id

    def canonical_id(self, name: str, type_: str, doc_id: str | None = None) -> str:
        scope = f"{doc_id}/" if (type_ in self.DOC_SCOPED_TYPES and doc_id) else ""

        if type_ == "FiscalPeriod":
            fp = normalize_fiscal_period(name)
            if fp:
                return f"FiscalPeriod:{fp}"
        if type_ in self.DOC_SCOPED_TYPES:
            sk = section_key(name)
            if sk:
                return f"{type_}:{scope}{sk}"

        k = norm_key(name)
        hit = self._by_key.get((type_, k))
        if hit and (not scope or hit.startswith(f"{type_}:{scope}")):
            return hit

        # Fuzzy match within type (and within doc scope for scoped types).
        pool = self._names_by_type.get(type_, {})
        if scope:
            pool = {n: i for n, i in pool.items() if i.startswith(f"{type_}:{scope}")}
        if pool and len(k) > 3:
            match = process.extractOne(k, list(pool.keys()), scorer=fuzz.token_sort_ratio)
            if match and match[1] >= self.threshold:
                return pool[match[0]]
        return f"{type_}:{scope}{slug(name)}"

    def resolve(self, name: str, type_: str, doc_id: str | None = None) -> str:
        eid = self.canonical_id(name, type_, doc_id)
        self.register(eid, type_, [name])
        return eid

    def lookup(self, mention: str, limit: int = 5, score_cutoff: float = 80.0) -> list[tuple[str, float]]:
        """Find existing entity ids for a free-text mention in a query (any type)."""
        k = norm_key(mention)
        results: dict[str, float] = {}
        fp = normalize_fiscal_period(mention)
        if fp:
            results[f"FiscalPeriod:{fp}"] = 100.0
        for type_, pool in self._names_by_type.items():
            if not pool:
                continue
            if k in pool:
                results[pool[k]] = max(results.get(pool[k], 0), 100.0)
                continue
            for name, score, _ in process.extract(k, list(pool.keys()), scorer=fuzz.token_sort_ratio, limit=limit):
                if score >= score_cutoff:
                    eid = pool[name]
                    results[eid] = max(results.get(eid, 0.0), float(score))
        return sorted(results.items(), key=lambda x: -x[1])[:limit]
