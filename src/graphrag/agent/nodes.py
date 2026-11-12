"""Nodes of the self-corrective state machine (CRAG + Self-RAG).

Each LLM-capable node has a deterministic fallback, so the loop runs
end-to-end with no API key: the server then returns *extractive*, cited,
verified evidence and lets the host agent (Claude Code) do the prose.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from ..indexing.embeddings import tokenize
from ..llm import LLM
from ..models import Evidence, GradeLabel, VerifyLabel
from ..retrieval.hybrid import HybridRetriever
from .sandbox import run_calculation
from .state import RAGState

log = logging.getLogger(__name__)

ABSTAIN_MSG = "Insufficient evidence in the indexed documents to verify this."

_QWORDS = set(
    "what which who whom whose when where why how much many value values does do did is are was were tell "
    "explain describe list give show find get please document documents doc spec specification mentioned say says "
    "according exactly current currently".split()
)
NEGATIONS = re.compile(r"\b(not|no|never|none|cannot|can't|won't|doesn't|don't|didn't|isn't|aren't|wasn't|without|"
                       r"excluded?|excluding|neither|nor|unchanged)\b", re.I)
NEG_WORDS = re.compile(r"\b(negative|loss|losses|decrease[ds]?|decline[ds]?|deficit|outflow|reduction|fell|dropped|minus)\b", re.I)
NUM_RE = re.compile(r"(?<![\w.])\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")
CITE_RE = re.compile(r"\[E(\d+)\]")

SYNONYMS = {
    "profit": ["net income", "earnings"], "income": ["earnings", "profit"], "revenue": ["net sales", "total revenue"],
    "sales": ["revenue"], "cost": ["expense", "expenses"], "expense": ["cost"], "limit": ["maximum", "max", "quota", "rate"],
    "limits": ["maximum", "quota", "rate limiting"], "throttle": ["rate limit", "429"], "timeout": ["timeout_ms", "deadline"],
    "latency": ["timeout", "ms"], "uptime": ["availability", "sla"], "sla": ["availability", "uptime"],
    "retry": ["retries", "backoff", "retry-after"], "retries": ["retry", "backoff"], "auth": ["authentication", "api key"],
    "key": ["api key", "credential"], "delete": ["remove", "DELETE"], "enterprise": ["enterprise tier", "override"],
    "override": ["supersedes", "overrides", "appendix"], "default": ["defaults to", "default limits"],
    "growth": ["increase", "change"], "margin": ["gross margin"], "debt": ["long-term debt", "borrowings"],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def key_terms(text: str) -> list[tuple[str, float]]:
    """Terms the evidence must contain, with weights (identifiers/numbers count double)."""
    terms: dict[str, float] = {}
    for m in re.finditer(r"`([^`]+)`|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b", text):
        terms[(m.group(1) or m.group(2)).lower()] = 2.0
    for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text):
        terms[n.replace(",", "")] = 1.5
    for t in re.findall(r"(?<![\w.`])[A-Za-z][A-Za-z0-9\-]+(?![\w.`])", text):
        tl = t.lower()
        if tl in _QWORDS or len(tl) < 3 or tl in terms:
            continue
        if tokenize(tl):
            terms.setdefault(tl, 1.0)
    return list(terms.items())


def _present(term: str, haystack: str, hay_tokens: set[str]) -> bool:
    if "_" in term or "." in term or term.isdigit() or re.fullmatch(r"\d+(\.\d+)?", term):
        return term in haystack
    if term in hay_tokens:
        return True
    stem = term[:5] if len(term) > 5 else term
    return any(t.startswith(stem) for t in hay_tokens) if len(term) > 4 else False


def coverage(question: str, evidence: list[Evidence]) -> tuple[float, list[str]]:
    terms = key_terms(question)
    if not terms:
        return (1.0 if evidence else 0.0), []
    hay = "\n".join(f"{e.chunk.header} {e.chunk.text}" for e in evidence).lower().replace(",", "")
    hay_tokens = set(re.findall(r"[a-z0-9_\-.]+", hay))
    got, total, missing = 0.0, 0.0, []
    for term, w in terms:
        total += w
        if _present(term, hay, hay_tokens):
            got += w
        else:
            missing.append(term)
    return got / total, missing


def sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        if para.lstrip().startswith("|"):
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`(])", para.replace("\n", " ")):
            s = s.strip()
            if len(s) > 15:
                out.append(s)
    return out


def _norm_num(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip("()%").lstrip("-")


def _content(text: str) -> set[str]:
    return {t for t in tokenize(CITE_RE.sub("", text)) if t not in _QWORDS and not t.isdigit()}


class Decomposition(BaseModel):
    sub_questions: list[str] = Field(description="1-4 self-contained sub-questions; just the original if it is atomic")


class GradeOut(BaseModel):
    label: str = Field(description="RELEVANT, AMBIGUOUS or MISSING")
    missing: list[str] = Field(default_factory=list, description="Specific facts/entities absent from the evidence")


class RewriteOut(BaseModel):
    queries: list[str] = Field(description="One rewritten search query per sub-question, using document terminology")


class VerifyOut(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    sign_or_negation_errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
class Nodes:
    def __init__(self, retriever: HybridRetriever, llm: LLM | None, settings) -> None:
        self.r = retriever
        self.llm = llm
        self.s = settings

    # 1. Router & decomposer ------------------------------------------------
    def route_and_decompose(self, state: RAGState) -> dict:
        q = state["query"].strip()
        subs: list[str] = []
        if self.llm is not None:
            out = self.llm.structured(
                "Split the user's question into the minimal set of self-contained sub-questions needed to answer it "
                "from reference documents. Do not add new questions. Keep identifiers verbatim.",
                q, Decomposition, max_tokens=400,
            )
            if out:
                subs = [s.strip() for s in out.sub_questions if s.strip()][:4]
        if not subs:
            parts = [p.strip() for p in re.split(r"\?|;|\band (?=(?:what|how|which|who|when|where|why|does|is|are)\b)", q, flags=re.I)]
            subs = [p.strip(" ,;:") for p in parts if len(p.split()) >= 3][:4] or [q]
        return {
            "sub_questions": subs, "search_queries": list(subs), "hops": self.s.graph_hops,
            "vector_k": self.s.vector_k, "round": 0, "regenerations": 0, "feedback": "",
            "trace": [f"route: {len(subs)} sub-question(s)"],
        }

    # 2. Hybrid retrieval ---------------------------------------------------
    def retrieve(self, state: RAGState) -> dict:
        merged: dict[str, Evidence] = {}
        facts: list[str] = []
        for sq in state["search_queries"]:
            res = self.r.retrieve(sq, state.get("filters"), hops=state["hops"], vector_k=state["vector_k"])
            for e in res.evidence:
                cur = merged.get(e.chunk.chunk_id)
                if cur is None or e.score > cur.score:
                    merged[e.chunk.chunk_id] = e
            facts.extend(res.graph_facts)
        cap = min(10, self.s.final_k * len(state["search_queries"]) + 2)
        evidence = sorted(merged.values(), key=lambda e: -e.score)[:cap]
        n = state["round"] + 1
        return {
            "evidence": evidence, "graph_facts": list(dict.fromkeys(facts))[:20], "round": n,
            "trace": [f"retrieve#{n}: {len(evidence)} chunks (hops={state['hops']}, k={state['vector_k']})"],
        }

    # 3. CRAG document grader ----------------------------------------------
    def grade_documents(self, state: RAGState) -> dict:
        ev = state["evidence"]
        cov, missing = {}, {}
        for sq in state["sub_questions"]:
            c, miss = coverage(sq, ev)
            cov[sq], missing[sq] = round(c, 3), miss
        worst = min(cov.values()) if cov else 0.0
        label = GradeLabel.relevant if worst >= 0.75 else GradeLabel.ambiguous if worst >= 0.4 else GradeLabel.missing

        if self.llm is not None and ev:
            out = self.llm.structured(
                "You grade retrieved evidence for a question (Corrective RAG). RELEVANT: the evidence contains every "
                "specific entity, period, value and condition needed. AMBIGUOUS: partially. MISSING: not at all. "
                "Judge only by the evidence text.",
                f"Question: {state['query']}\n\nEvidence:\n{_evidence_block(ev, 1200)}", GradeOut, max_tokens=300,
            )
            if out and out.label.upper() in {"RELEVANT", "AMBIGUOUS", "MISSING"}:
                llm_label = GradeLabel(out.label.upper())
                # Take the more conservative of heuristic and LLM verdicts.
                order = [GradeLabel.missing, GradeLabel.ambiguous, GradeLabel.relevant]
                label = min(label, llm_label, key=order.index)
                if out.missing:
                    missing[state["sub_questions"][0]] = list(dict.fromkeys(missing.get(state["sub_questions"][0], []) + out.missing))
        detail = ", ".join(f"{v:.0%}" for v in cov.values())
        return {"coverage": cov, "missing_terms": missing, "grade": label.value,
                "trace": [f"grade: {label.value} (coverage {detail})"]}

    def after_grade(self, state: RAGState) -> str:
        if state["grade"] == GradeLabel.relevant.value:
            return "generate"
        if state["round"] <= self.s.max_retries:
            return "rewrite"
        if state["grade"] == GradeLabel.ambiguous.value:
            return "generate"  # best effort, will be marked partially verified
        return "abstain"

    # 4. Query rewrite / hop expansion -------------------------------------
    def rewrite_and_expand(self, state: RAGState) -> dict:
        subs = state["sub_questions"]
        new_queries: list[str] = []
        if self.llm is not None:
            out = self.llm.structured(
                "Rewrite each search query so it matches how a formal specification, filing or contract would phrase it "
                "(e.g. 'profit' -> 'consolidated net income'). Keep identifiers verbatim. Return one query per input.",
                "\n".join(f"- {q} (missing: {', '.join(state['missing_terms'].get(sq, [])) or 'n/a'})"
                          for q, sq in zip(state["search_queries"], subs)),
                RewriteOut, max_tokens=400,
            )
            if out and len(out.queries) == len(subs):
                new_queries = out.queries
        if not new_queries:
            for q, sq in zip(state["search_queries"], subs):
                q = sq  # rebuild from the original wording so expansions don't compound
                extra: list[str] = []
                for term in state["missing_terms"].get(sq, []):
                    extra += SYNONYMS.get(term, [])
                    for eid, _ in self.r.resolver.lookup(term, limit=2, score_cutoff=75):
                        extra.append(eid.split(":", 1)[1].split("/")[-1].replace("_", " "))
                for t, _ in key_terms(sq):
                    extra += SYNONYMS.get(t, [])[:1]
                seen = set(tokenize(q))
                additions = []
                for x in dict.fromkeys(extra):
                    toks = set(tokenize(x))
                    if toks and not toks <= seen:
                        additions.append(x)
                        seen |= toks
                new_queries.append(" ".join([q] + additions))
        hops = min(state["hops"] + 1, self.s.max_graph_hops)
        k = min(state["vector_k"] + 10, 50)
        return {"search_queries": new_queries, "hops": hops, "vector_k": k,
                "trace": [f"rewrite: hops->{hops}, k->{k}; queries={new_queries}"]}

    # 5. Grounded generation + sandbox -------------------------------------
    def generate(self, state: RAGState) -> dict:
        ev = state["evidence"]
        citations = {f"E{i}": e.chunk.chunk_id for i, e in enumerate(ev, 1)}
        if self.llm is not None:
            draft, calcs = self._generate_llm(state)
            mode = "llm"
        else:
            draft, calcs = self._generate_extractive(state)
            mode = "extractive"
        return {"draft": draft, "citations": citations, "calculations": calcs,
                "trace": [f"generate ({mode}): {len(sentences(draft))} sentence(s), {len(calcs)} calc(s)"]}

    def _generate_llm(self, state: RAGState) -> tuple[str, list[dict]]:
        assert self.llm is not None
        system = (
            "Answer ONLY from the numbered evidence. After every sentence that states a fact, cite its source as [E#]. "
            "If evidence does not state something, say it is not stated; never guess. Preserve negations and accounting "
            "signs exactly: '(0.3)' means negative 0.3. When one section overrides another, give the overriding value "
            "and name both sections. For ANY arithmetic (differences, growth, ratios, conversions) do not compute it "
            "yourself: write <calc label=\"short label\">python arithmetic expression</calc>; available functions: "
            "pct_change(old,new), cagr(start,end,years), round, abs, min, max, avg. Reproduce tables as Markdown when "
            "the question asks for them. Be concise."
        )
        prompt = (f"Question: {state['query']}\n\nGraph facts:\n" + "\n".join(state.get("graph_facts", [])[:12]) +
                  f"\n\nEvidence:\n{_evidence_block(state['evidence'], 2500)}")
        if state.get("feedback"):
            prompt += f"\n\nYour previous draft failed verification. Fix these issues:\n{state['feedback']}"
        raw = self.llm.complete(system, prompt, max_tokens=1500)
        calcs: list[dict] = []

        def sub(m: re.Match) -> str:
            res = run_calculation(m.group(2), m.group(1) or "")
            calcs.append(res)
            return f"**{_fmt(res['result'])}**" if res["ok"] else f"[calculation error: {res['error']}]"

        draft = re.sub(r'<calc(?:\s+label="([^"]*)")?>(.*?)</calc>', sub, raw, flags=re.S)
        return draft.strip(), calcs

    def _generate_extractive(self, state: RAGState) -> tuple[str, list[dict]]:
        ev = state["evidence"]
        lines: list[str] = []
        used_tables: set[str] = set()
        for sq in state["sub_questions"]:
            terms = {t for t, _ in key_terms(sq)}
            scored: list[tuple[float, str, int]] = []
            for i, e in enumerate(ev, 1):
                if e.chunk.chunk_type == "table":
                    continue
                for s in sentences(e.chunk.text):
                    toks = set(re.findall(r"[a-z0-9_\-.]+", s.lower().replace(",", "")))
                    hit = sum(1 for t in terms if _present(t, s.lower(), toks))
                    if hit >= min(2, len(terms)):
                        bonus = 0.5 if "graph:MODIFIES" in e.sources else 0.0
                        scored.append((hit / max(1, len(terms)) + bonus + 0.1 * e.score, s, i))
            scored.sort(key=lambda x: -x[0])
            picked, seen = [], set()
            for sc, s, i in scored:
                if s in seen or sc < 0.25:
                    continue
                picked.append(f"- {s} [E{i}]")
                seen.add(s)
                if len(picked) >= 4:
                    break
            if len(state["sub_questions"]) > 1:
                lines.append(f"**{sq.rstrip('?')}?**")
            lines.extend(picked)
            # Relevant tables, reproduced intact.
            for i, e in enumerate(ev, 1):
                if e.chunk.chunk_type != "table" or e.chunk.chunk_id in used_tables:
                    continue
                c, _ = coverage(sq, [e])
                if c >= 0.5:
                    units = f" (units: {e.chunk.units})" if e.chunk.units else ""
                    table_only = "\n".join(l for l in e.chunk.text.splitlines() if l.strip().startswith("|"))
                    lines.append(f"\nTable from {e.chunk.citation}{units} [E{i}]:\n\n{_trim_table(table_only)}\n")
                    used_tables.add(e.chunk.chunk_id)
                    break
        overrides = [f for f in state.get("graph_facts", []) if "-[MODIFIES]->" in f]
        if overrides:
            lines.append("\n**Overrides detected in the document graph** (check which applies):")
            lines.extend(f"- {f}" for f in overrides[:4])
        if not any(l.startswith("- ") or l.lstrip().startswith("Table") for l in lines):
            return ABSTAIN_MSG, []
        return "\n".join(lines).strip(), []

    # 6. Self-RAG hallucination & negation verifier ------------------------
    def verify(self, state: RAGState) -> dict:
        draft = state["draft"]
        if draft.strip() == ABSTAIN_MSG:
            return {"verify": VerifyLabel.grounded.value, "unsupported": [], "trace": ["verify: abstention"]}
        cmap = state["citations"]
        chunks = {e.chunk.chunk_id: e.chunk for e in state["evidence"]}
        calc_values = {_norm_num(_fmt(c["result"])) for c in state.get("calculations", []) if c.get("ok")}
        unsupported, sign_errors = [], []

        for claim in _claims(draft):
            cited = [cmap.get(f"E{n}") for n in CITE_RE.findall(claim)]
            cited_text = "\n".join(chunks[c].text for c in cited if c in chunks)
            content = _content(claim)
            if not cited_text:
                if len(content) >= 4 and not re.search(r"not (stated|specified|mentioned)|insufficient", claim, re.I):
                    unsupported.append(claim)
                continue
            # lexical support: best matching sentence (or table) in cited chunks
            cands = sentences(cited_text) + [cited_text]
            best, best_ov = "", 0.0
            for s in cands:
                ov = len(content & _content(s)) / max(1, len(content))
                if ov > best_ov:
                    best, best_ov = s, ov
            if best_ov < 0.5:
                unsupported.append(claim)
                continue
            # numbers must appear in cited evidence or come from the sandbox
            src_nums = {_norm_num(n) for n in NUM_RE.findall(cited_text)}
            for n in NUM_RE.findall(CITE_RE.sub("", claim)):
                nn = _norm_num(n)
                if nn and nn not in src_nums and nn not in calc_values:
                    unsupported.append(claim)
                    break
                # parenthetical negative in source but presented as positive
                if nn and re.search(rf"\(\s*{re.escape(nn)}\s*\)", cited_text) and not (
                    n.startswith("(") or n.startswith("-") or NEG_WORDS.search(claim)
                ):
                    sign_errors.append(f"{claim}  <- source shows ({nn}), i.e. negative")
            # negation consistency against the best-supporting sentence
            if best is not cited_text and bool(NEGATIONS.search(claim)) != bool(NEGATIONS.search(best)):
                sign_errors.append(f"{claim}  <- negation differs from source: \"{best[:160]}\"")

        if self.llm is not None:
            out = self.llm.structured(
                "Check the answer against the evidence. List claims not supported by the evidence and any claim where "
                "a negation or accounting sign ((x) = negative) was inverted. Quote claims verbatim.",
                f"Answer:\n{draft}\n\nEvidence:\n{_evidence_block(state['evidence'], 2000)}", VerifyOut, max_tokens=500,
            )
            if out:
                unsupported += [c for c in out.unsupported_claims if c not in unsupported]
                sign_errors += [c for c in out.sign_or_negation_errors if c not in sign_errors]

        if sign_errors:
            label = VerifyLabel.sign_error
        elif unsupported:
            label = VerifyLabel.hallucinated
        else:
            label = VerifyLabel.grounded
        feedback = "\n".join([f"UNSUPPORTED: {c}" for c in unsupported] + [f"SIGN/NEGATION: {c}" for c in sign_errors])
        return {"verify": label.value, "unsupported": unsupported + sign_errors, "feedback": feedback,
                "trace": [f"verify: {label.value} ({len(unsupported)} unsupported, {len(sign_errors)} sign/negation)"]}

    def after_verify(self, state: RAGState) -> str:
        if state["verify"] == VerifyLabel.grounded.value:
            return "finalize"
        if self.llm is not None and state["regenerations"] < self.s.max_regenerations:
            return "regenerate"
        return "finalize"

    def bump_regeneration(self, state: RAGState) -> dict:
        return {"regenerations": state["regenerations"] + 1, "trace": ["regenerate with verifier feedback"]}

    # 7. Terminal nodes ------------------------------------------------------
    def abstain(self, state: RAGState) -> dict:
        near = [f"{e.chunk.citation}" for e in state.get("evidence", [])[:3]]
        missing = sorted({t for v in state.get("missing_terms", {}).values() for t in v})
        return {"final": {
            "status": "abstained", "answer": ABSTAIN_MSG,
            "missing": missing, "nearest_sources": near, "citations": [], "calculations": [],
            "unsupported_claims": [], "retrieval_rounds": state.get("round", 0),
        }, "trace": ["abstain: evidence missing after max retries"]}

    def finalize(self, state: RAGState) -> dict:
        draft = state["draft"]
        unsupported = state.get("unsupported", [])
        removed = []
        if unsupported:
            # Strip claims that failed verification rather than returning them.
            kept = []
            bad = [u.split("  <- ")[0].strip() for u in unsupported]
            for line in draft.splitlines():
                body = line.strip().lstrip("-* ").strip()
                if body and not body.startswith(("|", "Table from")) and any(b and b in body for b in bad):
                    removed.append(line.strip())
                    continue
                kept.append(line)
            draft = "\n".join(kept).strip() or ABSTAIN_MSG
        if draft == ABSTAIN_MSG:
            status = "abstained"
        elif state.get("grade") == GradeLabel.relevant.value and not unsupported:
            status = "verified"
        else:
            status = "partially_verified"
        used = set(CITE_RE.findall(draft))
        ev = state["evidence"]
        cites = [
            {"ref": f"E{i}", "chunk_id": e.chunk.chunk_id, "source": e.chunk.citation, "type": e.chunk.chunk_type,
             "retrieved_by": e.sources, **({"graph_path": e.graph_path} if e.graph_path else {})}
            for i, e in enumerate(ev, 1) if str(i) in used
        ]
        return {"final": {
            "status": status, "answer": draft, "citations": cites, "calculations": state.get("calculations", []),
            "unsupported_claims": unsupported, "removed_lines": removed, "graph_facts": state.get("graph_facts", [])[:8],
            "retrieval_rounds": state.get("round", 0), "coverage": state.get("coverage", {}),
        }, "trace": [f"finalize: {status}"]}


# ---------------------------------------------------------------------------
def _claims(draft: str) -> list[str]:
    out = []
    in_override_block = False
    for line in draft.splitlines():
        s = line.strip()
        if s.startswith("Table from"):  # tables are verbatim evidence, not generated claims
            continue
        if s.startswith("**Overrides detected"):
            in_override_block = True
            continue
        if not s or s.startswith("|") or s.startswith("#") or (s.startswith("**") and s.endswith("**")) or s.startswith("Table from"):
            continue
        if in_override_block and s.startswith("- ") and "-[" in s:
            continue
        s = s.lstrip("-* ").strip()
        out.extend(x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", s) if len(x) > 10)
    return out


def _evidence_block(evidence: list[Evidence], max_chars: int) -> str:
    parts = []
    for i, e in enumerate(evidence, 1):
        parts.append(f"[E{i}] ({e.chunk.citation}) {e.chunk.header}\n{e.chunk.text[:max_chars]}")
    return "\n\n".join(parts)


def _trim_table(md: str, max_rows: int = 30) -> str:
    lines = md.splitlines()
    return md if len(lines) <= max_rows + 2 else "\n".join(lines[: max_rows + 2] + [f"| ... {len(lines) - max_rows - 2} more rows |"])


def _fmt(x: float) -> str:
    return f"{x:,.4f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)
