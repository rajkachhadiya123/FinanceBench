"""
Compare two end-to-end answer runs question by question (all 150).

Each run is identified by its retrieval label in answer_results.jsonl. Its final
answer per question is the LAST record written with an accepted prompt version
(later records are the top-10 retry or a prompt-v2 re-run, which superseded the
earlier answer in that run).

    "qwen"        -> qwen+bm25+cohere-rerank          prompts v1 + v2 (v3 was removed)
    "qwen_headed" -> qwen_headed+bm25+cohere-rerank   prompt v2

Numeric answers are graded automatically (tools/test_answers.grade). Free-text
answers (REVIEW) and auto-WRONG answers need a human verdict: put them in
data/PROCESSED/manual_verdicts.json as {"<run>|<financebench_id>": "CORRECT" |
"PARTIAL" | "WRONG"}. A manual verdict is keyed to the exact answer text's hash,
so a changed answer is never silently given an old verdict.

Usage:
    python tools/compare_answer_runs.py              # summary + McNemar
    python tools/compare_answer_runs.py --pending    # answers still needing a verdict
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.test_answers import QUESTIONS_PATH, RESULTS_PATH, grade, is_refusal  # noqa: E402

RUNS = {
    "qwen": ("qwen+bm25+cohere-rerank", ("v1", "v2")),
    "qwen_headed": ("qwen_headed+bm25+cohere-rerank", ("v2",)),
}
MODEL = "gpt-oss-120b"
VERDICTS_PATH = ROOT / "data" / "PROCESSED" / "manual_verdicts.json"
OUTCOMES = ("CORRECT", "PARTIAL", "WRONG", "NOT FOUND", "PENDING")


def answer_hash(answer: str) -> str:
    return hashlib.md5(answer.encode("utf-8")).hexdigest()[:10]


def final_answers(label: str, prompts: tuple[str, ...]) -> dict[str, dict]:
    """Last v2 record if the run has one for that question, else last v1 record."""
    best: dict[str, dict] = {}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("prompt", "v1")
            if r.get("model") != MODEL or r.get("retrieval") != label or p not in prompts:
                continue
            prev = best.get(r["financebench_id"])
            if prev is None or prompts.index(p) >= prompts.index(prev.get("prompt", "v1")):
                best[r["financebench_id"]] = r
    return best


def outcome(run: str, qid: str, record: dict | None, manual: dict) -> str:
    if record is None or record.get("answer") in (None, "", "ERROR"):
        return "PENDING"
    if is_refusal(record["answer"]):
        return "NOT FOUND"
    verdict, _ = grade(record["expected"], record["answer"])
    if verdict == "CORRECT":
        return "CORRECT"
    m = manual.get(f"{run}|{qid}")
    if m and m["hash"] == answer_hash(record["answer"]):
        return m["verdict"]
    return "PENDING"


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    fixed = sum(1 for i in a if b[i] == "CORRECT" and a[i] != "CORRECT")
    broke = sum(1 for i in a if a[i] == "CORRECT" and b[i] != "CORRECT")
    n, m = fixed + broke, min(fixed, broke)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(m + 1)) / 2 ** n)
    return fixed, broke, p


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = {q["financebench_id"]: q for q in map(json.loads, f)}
    manual = json.loads(VERDICTS_PATH.read_text()) if VERDICTS_PATH.exists() else {}
    finals = {run: final_answers(*spec) for run, spec in RUNS.items()}
    outcomes = {run: {qid: outcome(run, qid, finals[run].get(qid), manual) for qid in questions}
                for run in RUNS}

    if "--pending" in sys.argv:
        for run in RUNS:
            for qid, o in outcomes[run].items():
                r = finals[run].get(qid)
                if o == "PENDING" and r:
                    print(json.dumps({"key": f"{run}|{qid}", "hash": answer_hash(r["answer"]),
                                      "question": questions[qid]["question"],
                                      "expected": r["expected"], "answer": r["answer"]},
                                     ensure_ascii=False))
        return

    print(f"{'outcome':10s} " + " ".join(f"{run:>14s}" for run in RUNS))
    for o in OUTCOMES:
        print(f"{o:10s} " + " ".join(
            f"{Counter(outcomes[run].values())[o]:>5d} ({Counter(outcomes[run].values())[o] / len(questions):4.0%}) "
            for run in RUNS))

    a, b = outcomes["qwen"], outcomes["qwen_headed"]
    fixed, broke, p = mcnemar(a, b)
    print(f"\nqwen_headed vs qwen (correct): +{fixed} / -{broke}, McNemar p = {p:.2f}")

    by_type = defaultdict(list)
    for qid, q in questions.items():
        by_type[q["question_type"]].append(qid)
    print("\nCorrect by question type:")
    for t, ids in sorted(by_type.items()):
        print(f"  {t:18s} n={len(ids):3d}: " + " | ".join(
            f"{run} {sum(outcomes[run][i] == 'CORRECT' for i in ids)}" for run in RUNS))

    print("\nQuestions whose outcome changed (qwen -> qwen_headed):")
    for qid in sorted(questions):
        if a[qid] != b[qid]:
            print(f"  {qid}: {a[qid]} -> {b[qid]}")


if __name__ == "__main__":
    main()
