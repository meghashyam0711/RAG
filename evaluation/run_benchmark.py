"""Ablation benchmark: Naive RAG vs Hybrid RAG vs Self-Corrective GraphRAG.

Configurations
  naive   - dense vector search only, top-5, no graph, no correction loop
  hybrid  - dense + BM25 + RRF + rerank, no graph, no correction loop
  full    - complete pipeline (graph leg, CRAG retries, verification)

Metrics
  context_recall@5   gold evidence page (same doc) appears in the retrieved chunks
  numeric_accuracy   every number in the gold answer appears in the prediction (1% tolerance)
  abstain_rate       fraction answered "insufficient evidence"
  p50 / p95 latency  milliseconds per question

Datasets
  --dataset financebench   PatronusAI/financebench from Hugging Face (pip install '.[eval]').
                           Put the filings in --pdf-dir as <doc_name>.pdf (download from each row's doc_link).
  --dataset jsonl          Your own file: one JSON object per line with
                           {"question", "answer", "doc": "<path>", "evidence_pages": [int], "domain": "finance"}

Example
  python evaluation/run_benchmark.py --dataset financebench --pdf-dir data/raw_docs --limit 150 --out results.json
  python evaluation/run_benchmark.py --dataset jsonl --questions evaluation/sample_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrag.agent.nodes import ABSTAIN_MSG  # noqa: E402
from graphrag.engine import GraphRAGEngine  # noqa: E402

NUM = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?")


def load_financebench(pdf_dir: Path, limit: int | None) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    rows = load_dataset("PatronusAI/financebench", split="train")
    out = []
    for r in rows:
        pdf = pdf_dir / f"{r['doc_name']}.pdf"
        if not pdf.exists():
            continue
        pages = [int(e.get("evidence_page_num", -1)) + 1 for e in (r.get("evidence") or []) if e.get("evidence_page_num") is not None]
        out.append({"question": r["question"], "answer": str(r["answer"]), "doc": str(pdf), "evidence_pages": pages, "domain": "finance"})
        if limit and len(out) >= limit:
            break
    if not out:
        raise SystemExit(f"No FinanceBench PDFs found in {pdf_dir}. Save filings as <doc_name>.pdf.")
    return out


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        r["doc"] = str((path.parent / r["doc"]).resolve()) if not Path(r["doc"]).is_absolute() else r["doc"]
    return rows[:limit] if limit else rows


def _nums(text: str) -> list[float]:
    out = []
    for n in NUM.findall(text):
        neg = n.startswith("(") or n.startswith("-")
        try:
            v = float(re.sub(r"[,$()\-]", "", n))
        except ValueError:
            continue
        out.append(-v if neg else v)
    return out


def numeric_match(gold: str, pred: str) -> bool | None:
    g = _nums(gold)
    if not g:
        return None
    p = _nums(pred)
    return all(any(abs(x - y) <= max(1e-9, 0.01 * abs(x)) for y in p) for x in g)


def run_config(engine: GraphRAGEngine, name: str, q: dict) -> dict:
    t0 = time.perf_counter()
    doc_id = engine.resolve_doc_filter(Path(q["doc"]).stem)
    if name == "full":
        res = engine.ask(q["question"], Path(q["doc"]).stem)
        pred = res["answer"]
        pages = [_page(engine, c["chunk_id"]) for c in res.get("citations", [])]
        if not pages:  # abstained / no citations: score recall on what was retrieved
            pages = [p for p in (res.get("nearest_sources") or [])]
    else:
        use_sparse = name == "hybrid"
        r = engine.retriever.retrieve(q["question"], doc_id, hops=0, use_sparse=use_sparse)
        if name == "naive":  # pure dense ranking, no rerank
            dense = engine.vectors.dense_search(q["question"], 5, doc_id)
            chunks = [c for c, _ in dense]
        else:
            chunks = [e.chunk for e in r.evidence]
        pred = "\n".join(c.text for c in chunks)
        pages = [(c.page_start, c.page_end) for c in chunks]
    ms = (time.perf_counter() - t0) * 1000
    gold_pages = set(q.get("evidence_pages") or [])
    recall = None
    if gold_pages:
        recall = any(isinstance(p, tuple) and p[0] is not None and any(p[0] <= g <= (p[1] or p[0]) for g in gold_pages)
                     for p in pages)
    return {"config": name, "question": q["question"], "recall": recall, "numeric": numeric_match(q["answer"], pred),
            "abstained": ABSTAIN_MSG in pred, "latency_ms": ms}


def _page(engine, chunk_id):
    c = engine.vectors.get_chunks([chunk_id])
    return (c[0].page_start, c[0].page_end) if c else (None, None)


def summarize(rows: list[dict]) -> dict:
    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "n": len(rows),
        "context_recall@5": rate("recall"),
        "numeric_accuracy": rate("numeric"),
        "abstain_rate": rate("abstained"),
        "p50_latency_ms": round(statistics.median(lat), 1) if lat else None,
        "p95_latency_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["financebench", "jsonl"], required=True)
    ap.add_argument("--questions", type=Path, help="JSONL file for --dataset jsonl")
    ap.add_argument("--pdf-dir", type=Path, default=ROOT / "data" / "raw_docs")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--configs", default="naive,hybrid,full")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    qs = load_financebench(args.pdf_dir, args.limit) if args.dataset == "financebench" else load_jsonl(args.questions, args.limit)
    engine = GraphRAGEngine()
    for doc in sorted({(q["doc"], q.get("domain", "general")) for q in qs}):
        print(f"ingest {Path(doc[0]).name}: {engine.ingest(doc[0], doc[1])['status']}", file=sys.stderr)

    results, per_q = {}, []
    for cfg in args.configs.split(","):
        rows = [run_config(engine, cfg, q) for q in qs]
        per_q += rows
        results[cfg] = summarize(rows)

    print(f"\n{'config':<8} {'recall@5':>9} {'numeric':>8} {'abstain':>8} {'p50 ms':>8} {'p95 ms':>8}")
    for cfg, m in results.items():
        f = lambda v: "-" if v is None else f"{v}"
        print(f"{cfg:<8} {f(m['context_recall@5']):>9} {f(m['numeric_accuracy']):>8} {f(m['abstain_rate']):>8} "
              f"{f(m['p50_latency_ms']):>8} {f(m['p95_latency_ms']):>8}")
    if args.out:
        args.out.write_text(json.dumps({"summary": results, "per_question": per_q}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
