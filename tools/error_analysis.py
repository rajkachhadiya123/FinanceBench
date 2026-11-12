"""
Error analysis of the 79.3% run (headed retrieval, prompt v2): for every answer that
is not CORRECT, was a gold evidence page among the chunks the LLM actually saw?

    retrieval miss            no gold evidence page in the chunks
    partial retrieval         some, but not all, gold pages (multi-page questions)
    retrieved, answered wrong every gold page was in the chunks

The chunks are rebuilt exactly as evaluate_guardrails.py does (same filter, dense +
BM25, RRF, 40 candidates, then the answer's top_k: 5, or 10 after a NOT FOUND retry),
using ONLY cached Cohere scores. A question whose scores are not cached is reported
as "uncached"; this script never calls an API.

A chunk covers a gold page when it is from the evidence filing and
page_start <= evidence_page_num <= page_end (both 0-based PDF indexes).

Usage:
    python tools/error_analysis.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

import compare_answer_runs as runs  # noqa: E402
from evaluate_reranker import no_llm_split  # noqa: E402
from source.RETRIEVAL.retriever import RERANK_CANDIDATES, Retriever  # noqa: E402

EMBEDDING = "qwen_headed"
CACHE_PATH = ROOT / "data" / "PROCESSED" / "rerank_cache_rerank-v4.0-pro_qwen_headed.json"


def rebuild(retriever: Retriever, cache: dict, qid: str, question: str, top_k: int) -> list[dict] | None:
    allowed = retriever._allowed_positions(question)
    fused = retriever._rrf_fuse([retriever._dense_search(question, 20, allowed),
                                 retriever._bm25_search(question, 20, allowed)])
    candidates = fused[:RERANK_CANDIDATES["cohere"]]
    key = f"{qid}:{','.join(map(str, candidates))}"
    if key not in cache:
        return None  # would need a Cohere call: skip
    scores = dict(zip(candidates, cache[key]))
    ordered = sorted(candidates, key=lambda p: -scores[p]) + fused[len(candidates):]
    return [retriever._build_result(pos, rank) for rank, pos in enumerate(ordered[:top_k], 1)]


def gold_coverage(chunks: list[dict], evidence: list[dict]) -> tuple[int, int]:
    gold = {(e["doc_name"], int(e["evidence_page_num"])) for e in evidence}
    hit = {g for g in gold for c in chunks
           if c["doc_name"] == g[0] and c["page_start"] <= g[1] <= c["page_end"]}
    return len(hit), len(gold)


def main():
    with open(runs.QUESTIONS_PATH, encoding="utf-8") as f:
        questions = {q["financebench_id"]: q for q in map(json.loads, f)}
    final = runs.final_answers(*runs.RUNS[EMBEDDING])
    manual = json.loads(runs.VERDICTS_PATH.read_text())
    split = no_llm_split(list(questions.values()))
    no_llm = set(split["tune"] + split["holdout"])
    cache = json.loads(CACHE_PATH.read_text())
    retriever = Retriever(embedding=EMBEDDING)

    rows = []
    for qid in sorted(questions):
        rec, q = final[qid], questions[qid]
        outcome = runs.outcome(EMBEDDING, qid, rec, manual)
        chunks = rebuild(retriever, cache, qid, q["question"], rec["top_k"])
        if chunks is None:
            bucket, hit, total = "uncached", None, len(q["evidence"])
        else:
            hit, total = gold_coverage(chunks, q["evidence"])
            bucket = ("retrieval miss" if hit == 0 else
                      "partial retrieval" if hit < total else "all gold pages retrieved")
        rows.append({"qid": qid, "outcome": outcome, "top_k": rec["top_k"], "bucket": bucket,
                     "gold_hit": hit, "gold_total": total, "no_llm": qid in no_llm,
                     "type": q["question_type"]})
    retriever.close()

    print(f"Outcomes: {dict(Counter(r['outcome'] for r in rows))}\n")
    print("Gold evidence page(s) among the chunks the LLM saw, by outcome:")
    table = defaultdict(Counter)
    for r in rows:
        table[r["outcome"]][r["bucket"]] += 1
    buckets = ["all gold pages retrieved", "partial retrieval", "retrieval miss", "uncached"]
    print(f"  {'outcome':10s} " + " | ".join(f"{b:>24s}" for b in buckets))
    for o in ("CORRECT", "PARTIAL", "WRONG", "NOT FOUND"):
        print(f"  {o:10s} " + " | ".join(f"{table[o][b]:>24d}" for b in buckets))

    failed = [r for r in rows if r["outcome"] != "CORRECT"]
    print(f"\nNot correct: {len(failed)}  ->  {dict(Counter(r['bucket'] for r in failed))}")
    for label, group in (("answer in filing (71)", [r for r in failed if r["no_llm"]]),
                         ("needs calculation (79)", [r for r in failed if not r["no_llm"]])):
        print(f"  {label:24s} n={len(group):2d}: {dict(Counter(r['bucket'] for r in group))}")
    print("\nPer failed question:")
    for r in failed:
        print(f"  {r['qid']}  {r['outcome']:9s} top_k={r['top_k']:2d}  gold {r['gold_hit']}/{r['gold_total']}"
              f"  {'no-LLM' if r['no_llm'] else 'calc  '}  {r['type']}")


if __name__ == "__main__":
    main()
