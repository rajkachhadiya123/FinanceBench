"""
Measure Cohere reranking on top of the Qwen hybrid retriever.

Question set: the 71 questions whose expected answer is literally written in
the filing (strict matching), split 43 tune / 28 holdout (seed 42, stratified
by question type) -- the same set as documentation/experiment_log V2/V3.

Per question: filter -> Qwen dense top 20 + BM25 top 20 -> RRF (up to 40
candidates) -> one Cohere call scores all of them. Because a cross-encoder
scores each (question, chunk) pair independently, the same scores also give the
"rerank only the top 20" variant without a second call.

Metric: rank of the first chunk that is from a labelled filing AND contains
the expected answer (strict: years never count as answer values; unit scaling
only when the answer states a unit).

Scores are cached in data/PROCESSED/rerank_cache_<model>.json, so reruns and
interruptions never spend the trial key's 1,000-calls/month quota twice.

Usage:
    python tools/evaluate_reranker.py                     # rerank-v4.0-pro
    python tools/evaluate_reranker.py rerank-v4.0-fast
    python tools/evaluate_reranker.py --embedding qwen_headed   # chunks with context headers

Headed variants are graded on the chunk text WITHOUT its header line, so a
header's page or note number ("page 55", "Note 8") can never count as an
answer value. Per-question ranks are saved to retrieval_ranks_<embedding>_<model>.json;
a non-default embedding is then compared question by question with "qwen".
"""
import json
import random
import re
import sys
import time
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_retrieval as ev  # noqa: E402
from source.RETRIEVAL.cohere_reranker import CohereReranker  # noqa: E402
from source.RETRIEVAL.retriever import Retriever  # noqa: E402

ARGS = sys.argv[1:]
EMBEDDING = ARGS.pop(ARGS.index("--embedding") + 1) if "--embedding" in ARGS else "qwen"
ARGS = [a for a in ARGS if a != "--embedding"]
MODEL = ARGS[0] if ARGS else "rerank-v4.0-pro"
BASELINE_EMBEDDING = "qwen"
HEADED = EMBEDDING.endswith("_headed")
PROCESSED = ROOT / "data" / "PROCESSED"
# The reranker reads the chunk text, which differs per embedding variant, so each
# variant keeps its own cache (the original "qwen" cache keeps its old name).
CACHE_PATH = PROCESSED / (f"rerank_cache_{MODEL}.json" if EMBEDDING == BASELINE_EMBEDDING
                          else f"rerank_cache_{MODEL}_{EMBEDDING}.json")
RANKS_PATH = lambda emb: PROCESSED / f"retrieval_ranks_{emb}_{MODEL}.json"  # noqa: E731
CALL_SPACING = 6.5  # trial key: 10 calls/minute
UNIT_RE = re.compile(r"\$|\b(million|billion|thousand|mn|bn|usd)\b", re.IGNORECASE)


# --- strict answer matching -------------------------------------------------------
def answer_numbers(answer: str) -> set[float]:
    text = re.sub(r"COVID-19", "", answer, flags=re.IGNORECASE)
    return {n for n in ev.extract_numbers(text)
            if not (n.is_integer() and 1990 <= abs(n) <= 2035)}


def strict_in_text(answer: str, text: str) -> bool:
    nums = answer_numbers(answer)
    if not nums:
        return ev.keyword_ratio(answer, text) >= ev.KEYWORD_PASS_RATIO
    candidates = ev.extract_numbers(text)
    if UNIT_RE.search(answer):
        return any(ev.number_present(n, candidates) for n in nums)
    return any(abs(n - c) <= 0.01 * max(abs(n), abs(c), 1e-9) for n in nums for c in candidates)


def no_llm_split(questions: list[dict]) -> dict[str, list[str]]:
    by_type = defaultdict(list)
    for q in questions:
        evidence = "\n".join(e["evidence_text"] for e in q["evidence"])
        if strict_in_text(str(q["answer"]), evidence):
            by_type[q["question_type"]].append(q["financebench_id"])
    rng = random.Random(42)
    split = {"tune": [], "holdout": []}
    for ids in by_type.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        k = round(len(ids) * 0.6)
        split["tune"] += ids[:k]
        split["holdout"] += ids[k:]
    return split


def graded_text(text: str) -> str:
    """The text the answer is matched against: a headed chunk minus its header line."""
    return text.split("\n", 1)[1] if HEADED and text.startswith("[") else text


def grounded_rank(q: dict, texts_in_order: list[tuple[str, str]]) -> int | None:
    docs = {e["doc_name"] for e in q["evidence"]}
    for i, (doc, text) in enumerate(texts_in_order, 1):
        if doc in docs and strict_in_text(str(q["answer"]), text):
            return i
    return None


def hits_at(ranks: dict, k: int) -> int:
    return sum(1 for r in ranks.values() if r is not None and r <= k)


def mcnemar(a: dict, b: dict, k: int) -> tuple[int, int, float]:
    ok = lambda r: r is not None and r <= k
    fixed = sum(1 for i in a if ok(b[i]) and not ok(a[i]))
    broke = sum(1 for i in a if ok(a[i]) and not ok(b[i]))
    n, m = fixed + broke, min(fixed, broke)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(m + 1)) / 2 ** n)
    return fixed, broke, p


def main():
    with open(ev.QUESTIONS_PATH, encoding="utf-8") as f:
        questions = {q["financebench_id"]: q for q in map(json.loads, f)}
    split = no_llm_split(list(questions.values()))
    ids = split["tune"] + split["holdout"]
    print(f"No-LLM questions: {len(ids)} (tune {len(split['tune'])}, holdout {len(split['holdout'])})")

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    retriever = Retriever(embedding=EMBEDDING)
    reranker = CohereReranker(MODEL)
    print(f"Embedding: {EMBEDDING} | reranker: {MODEL} | cached questions: {len(cache)}\n")

    base, rr20, rr40, api_secs = {}, {}, {}, []
    for n, qid in enumerate(ids, 1):
        q = questions[qid]
        allowed = retriever._allowed_positions(q["question"])
        fused = retriever._rrf_fuse([
            retriever._dense_search(q["question"], 20, allowed),
            retriever._bm25_search(q["question"], 20, allowed),
        ])[:40]
        chunk = {p: (retriever.meta[p]["doc_name"], graded_text(retriever._chunk_text(p))) for p in fused}
        base[qid] = grounded_rank(q, [chunk[p] for p in fused])

        key = f"{qid}:{','.join(map(str, fused))}"  # cache valid only for this exact pool
        if key not in cache:
            if api_secs:
                time.sleep(CALL_SPACING)
            t = time.time()
            cache[key] = reranker.score(q["question"], [retriever._rerank_document(p) for p in fused])
            api_secs.append(time.time() - t)
            CACHE_PATH.write_text(json.dumps(cache))
        scores = dict(zip(fused, cache[key]))
        top20 = sorted(fused[:20], key=lambda p: -scores[p]) + fused[20:]
        top40 = sorted(fused, key=lambda p: -scores[p])
        rr20[qid] = grounded_rank(q, [chunk[p] for p in top20])
        rr40[qid] = grounded_rank(q, [chunk[p] for p in top40])
        if n % 10 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)} questions", flush=True)
    retriever.close()
    reranker.close()

    configs = {f"{EMBEDDING} hybrid (no rerank)": base, f"+ {MODEL} over top 20": rr20,
               f"+ {MODEL} over top 40": rr40}
    RANKS_PATH(EMBEDDING).write_text(json.dumps({"no_rerank": base, "rerank20": rr20, "rerank40": rr40}))
    for name, sel in (("ALL", ids), ("tune", split["tune"]), ("holdout", split["holdout"])):
        print(f"\n== {name} ({len(sel)}) ==")
        for label, ranks in configs.items():
            sub = {i: ranks[i] for i in sel}
            mrr = sum(1 / r for r in sub.values() if r) / len(sel)
            print(f"  {label:38s} " + " | ".join(
                f"@{k} {hits_at(sub, k):2d} ({hits_at(sub, k) / len(sel):.0%})" for k in (1, 3, 5, 10))
                + f" | MRR {mrr:.3f}")
    print("\n== significance vs no rerank (all) ==")
    for label, ranks in list(configs.items())[1:]:
        for k in (1, 5):
            f_, b_, p = mcnemar(base, ranks, k)
            print(f"  {label:38s} @{k}: +{f_}/-{b_}  p={p:.3f}")
    print("\n== by question type (all, @5) ==")
    by_type = defaultdict(list)
    for i in ids:
        by_type[questions[i]["question_type"]].append(i)
    for t, sel in sorted(by_type.items()):
        print(f"  {t:18s} n={len(sel):2d}: " + " | ".join(
            f"{label.split(' over ')[-1] if 'over' in label else 'no rerank'} "
            f"{hits_at({i: r[i] for i in sel}, 5)}" for label, r in configs.items()))
    if api_secs:
        print(f"\nCohere API time: avg {sum(api_secs) / len(api_secs):.2f}s per question "
              f"({len(api_secs)} new calls)")
    compare_with_baseline({"no_rerank": base, "rerank20": rr20, "rerank40": rr40}, split)


def compare_with_baseline(ranks: dict, split: dict):
    """Question-by-question comparison with the "qwen" run (its saved ranks file)."""
    if EMBEDDING == BASELINE_EMBEDDING or not RANKS_PATH(BASELINE_EMBEDDING).exists():
        return
    old = json.loads(RANKS_PATH(BASELINE_EMBEDDING).read_text())
    print(f"\n== {EMBEDDING} vs {BASELINE_EMBEDDING}, same pipeline (fixed / broken, McNemar) ==")
    for cfg in ("no_rerank", "rerank40"):
        for name, sel in (("all", split["tune"] + split["holdout"]), ("tune", split["tune"]),
                          ("holdout", split["holdout"])):
            a = {i: old[cfg][i] for i in sel}
            b = {i: ranks[cfg][i] for i in sel}
            line = []
            for k in (1, 5, 10):
                f_, b_, p = mcnemar(a, b, k)
                line.append(f"@{k} {hits_at(a, k)}->{hits_at(b, k)} (+{f_}/-{b_}, p={p:.2f})")
            print(f"  {cfg:9s} {name:7s} n={len(sel):2d}: " + " | ".join(line))
    for cfg in ("rerank40",):
        in_top5 = lambda r: r is not None and r <= 5  # noqa: E731
        changed = [i for i in old[cfg] if in_top5(old[cfg][i]) != in_top5(ranks[cfg][i])]
        if changed:
            print(f"\n  {cfg} questions that changed at top 5 (old rank -> new rank):")
            for i in sorted(changed):
                print(f"    {i}: {old[cfg][i]} -> {ranks[cfg][i]}")


if __name__ == "__main__":
    main()
