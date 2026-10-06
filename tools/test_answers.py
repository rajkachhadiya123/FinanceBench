"""
End-to-end smoke test: retrieval -> LLM (Groq) -> answer, printed side by side
against the expected answer.

With --computed-only it selects questions whose answer is NOT written in their
own labelled evidence -- the ones requiring calculation or synthesis. Those are
both the hardest case and the ones retrieval alone can never satisfy, so they
isolate whether the model can actually do the work.

Usage:
    python tools/test_answers.py                    # 10 questions, top_k=5
    python tools/test_answers.py --n 15 --top-k 15
    python tools/test_answers.py --computed-only --include-holdout --n 200
                                                    # every LLM-dependent question

Free-tier quota: answers are cached as they arrive, and the run stops cleanly
when the daily quota is spent. Rerun the same command the next day to continue.

The sample is a deterministic prefix of one shuffled ordering, so a run with
--n 10 tests a subset of --n 15. That makes results comparable across runs --
otherwise a changed score could come from a different question set rather than
from the setting being tested.
"""
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.GENERATION.answer_generator import AnswerGenerator, DailyLimitReached  # noqa: E402
from source.RETRIEVAL.retriever import Retriever  # noqa: E402
from tools.evaluate_retrieval import (  # noqa: E402
    KEYWORD_PASS_RATIO,
    extract_numbers,
    keyword_ratio,
    number_present,
)

QUESTIONS_PATH = ROOT / "data" / "EVAL" / "financebench_open_source.jsonl"
# Each answer is appended here as soon as it arrives, so an interrupted run
# keeps its work and a rerun skips questions already answered. Rate limits make
# discarding completed answers the expensive mistake.
RESULTS_PATH = ROOT / "data" / "PROCESSED" / "answer_results.jsonl"
DEFAULT_TOP_K = 5
REQUEST_SPACING = 5  # minimum seconds between questions
# Groq free tier: 8000 tokens/minute. After each call, wait long enough that the
# tokens it used fit inside the per-minute budget, instead of hitting 429s.
TOKENS_PER_MINUTE = 8000
RETRY_TOP_K = 10  # on a NOT FOUND, ask again once with this many chunks


def is_refusal(answer: str) -> bool:
    return answer.strip().upper().startswith("NOT FOUND")


def load_cached_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    cached = {}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially-written final line
            cached[record["financebench_id"]] = record
    return cached


def append_result(record: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


HOLDOUT_SIZE = 50  # locked away, never tuned against


def load_questions(computed_only: bool, include_holdout: bool = False) -> list[dict]:
    """
    Returns the dev split by default. The holdout is fixed by seed 1234 and
    excluded, so repeated tuning against dev cannot quietly inflate the final
    reported number.
    """
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    if not include_holdout:
        ordered = sorted(questions, key=lambda q: q["financebench_id"])
        random.Random(1234).shuffle(ordered)
        holdout_ids = {q["financebench_id"] for q in ordered[:HOLDOUT_SIZE]}
        questions = [q for q in questions if q["financebench_id"] not in holdout_ids]

    if computed_only:
        questions = [q for q in questions if not answer_written_in_evidence(q)]
    return questions


UNIT_RE = re.compile(r"\$|\b(million|billion|thousand|mn|bn|usd)\b", re.IGNORECASE)


def answer_written_in_evidence(record: dict) -> bool:
    """
    Is the expected answer literally present in the labelled evidence? If not,
    only an LLM (calculation or synthesis) can produce it.

    Stricter than evaluate_retrieval.answer_is_retrievable, which counted 13
    LLM-dependent questions as retrievable because (1) years in the answer
    ("in FY 2022") matched any year in the evidence, (2) "COVID-19" parsed as
    -19, and (3) unit scaling let a ratio like 0.83 match 833,033. Here years
    are ignored and scaling applies only when the answer states a unit.
    """
    answer = str(record["answer"])
    evidence = "\n".join(e["evidence_text"] for e in record["evidence"])
    expected = meaningful_numbers(re.sub(r"COVID-19", "", answer, flags=re.IGNORECASE))
    if not expected:
        return keyword_ratio(answer, evidence) >= KEYWORD_PASS_RATIO
    found = extract_numbers(evidence)
    if UNIT_RE.search(answer):
        return any(number_present(n, found) for n in expected)
    return any(abs(n - c) <= 0.01 * max(abs(n), abs(c), 1e-9) for n in expected for c in found)


def stratified_sample(questions: list[dict], n: int) -> list[dict]:
    """Even coverage across question_type, so one category can't dominate the score."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q)

    for group in by_type.values():
        random.Random(42).shuffle(group)

    sample, index = [], 0
    types = sorted(by_type)
    while len(sample) < min(n, len(questions)):
        group = by_type[types[index % len(types)]]
        slot = index // len(types)
        if slot < len(group):
            sample.append(group[slot])
        index += 1
        if index > n * len(types) + len(types):
            break
    return sample[:n]


def meaningful_numbers(text: str) -> set[float]:
    """
    Numbers worth grading against, excluding bare 4-digit years.

    Without this, an answer like "As of FY2022, Pepsico operates in North
    America, Latin America..." was treated as a numeric question: the grader
    compared 2022 against "more than 200 countries" and marked a correct
    answer WRONG. A year mentioned in a sentence is context, not the answer.
    """
    numbers = extract_numbers(text)
    return {
        n for n in numbers
        if not (n.is_integer() and 1990 <= n <= 2030 and f"{int(n)}" in text)
    }


def keyword_overlap(expected: str, generated: str) -> float:
    """Rough similarity hint for free-text answers, to assist human review."""
    stop = {"that", "this", "with", "the", "and", "for", "not", "was", "are", "were"}
    words = {w for w in re.findall(r"[a-z]{4,}", expected.lower()) if w not in stop}
    if not words:
        return 0.0
    lowered = generated.lower()
    return sum(1 for w in words if w in lowered) / len(words)


def grade(expected: str, generated: str) -> tuple[str, str]:
    """
    Returns (verdict, note). Only genuinely numeric answers are graded
    automatically; free-text answers are flagged for human judgement rather
    than trusted to an unvalidated automatic judge.
    """
    if not generated or generated == "ERROR":
        return "ERROR", "no answer produced"
    if generated.strip().upper().startswith("NOT FOUND"):
        return "REFUSED", "model declined to answer"

    expected_numbers = meaningful_numbers(expected)
    if not expected_numbers:
        overlap = keyword_overlap(expected, generated)
        return "REVIEW", f"free-text -- judge by eye (keyword overlap {overlap:.0%})"

    generated_numbers = extract_numbers(generated)
    if not generated_numbers:
        return "WRONG", "expected a number, none found in answer"

    if any(number_present(n, generated_numbers) for n in expected_numbers):
        return "CORRECT", ""
    return "WRONG", f"expected {sorted(expected_numbers)} got {sorted(generated_numbers)}"


def safe(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def main():
    n = 10
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    top_k = DEFAULT_TOP_K
    if "--top-k" in sys.argv:
        top_k = int(sys.argv[sys.argv.index("--top-k") + 1])
    computed_only = "--computed-only" in sys.argv
    include_holdout = "--include-holdout" in sys.argv
    targeted = "--no-targeted" not in sys.argv
    retry_top_k = RETRY_TOP_K  # 0 disables the wider retry on refusal
    if "--retry-top-k" in sys.argv:
        retry_top_k = int(sys.argv[sys.argv.index("--retry-top-k") + 1])

    questions = load_questions(computed_only, include_holdout)
    sample = stratified_sample(questions, n)

    kind = "COMPUTED-ANSWER ONLY" if computed_only else "ALL TYPES"
    scope = ("all 150, holdout INCLUDED" if include_holdout
             else f"dev only, holdout of {HOLDOUT_SIZE} excluded")
    print(f"Pool: {len(questions)} questions ({kind}; {scope}).")
    print(f"Testing {len(sample)}, top_k={top_k}, targeted={targeted}.")
    by_type = defaultdict(int)
    for q in sample:
        by_type[q["question_type"]] += 1
    print(f"Sample mix: {dict(by_type)}\n")

    if "--fresh" in sys.argv and RESULTS_PATH.exists():
        RESULTS_PATH.unlink()
    generator = AnswerGenerator()
    # Only reuse answers produced by the same model; older records (e.g. from
    # the earlier Gemini setup, which carry no "model" field) are ignored.
    cached = {qid: r for qid, r in load_cached_results().items()
              if r.get("model") == generator.model}
    if cached:
        print(f"Resuming: {len(cached)} {generator.model} answers already on disk will be reused.\n")

    retriever = Retriever(use_reranker=False)
    print(f"Model: {generator.model}\n")

    verdicts = []
    by_type_verdicts: dict[str, list[str]] = defaultdict(list)
    total_in = total_out = 0
    pacing = {"called": 0, "last_tokens": 0}
    stopped_early = False
    escalated = rescued = 0

    def ask(record: dict, k: int) -> tuple[dict, list[dict]]:
        """Retrieve top-k, generate, and cache the answer immediately."""
        # Pace by tokens: the per-minute token budget, not request count,
        # is what the free tier runs out of first.
        if pacing["called"]:
            time.sleep(max(REQUEST_SPACING, pacing["last_tokens"] / TOKENS_PER_MINUTE * 60))
        chunks = retriever.search(
            record["question"], top_k=k, metadata_filter=True, targeted=targeted
        )
        result = generator.generate(record["question"], chunks)
        pacing["called"] += 1
        pacing["last_tokens"] = result["input_tokens"] + result["output_tokens"]
        append_result({
            "financebench_id": record["financebench_id"],
            "question_type": record["question_type"],
            "expected": str(record["answer"]),
            "retrieved_docs": sorted({c["doc_name"] for c in chunks}),
            "truth_doc": record["doc_name"],
            "model": generator.model,
            "top_k": k,
            **{key: result[key] for key in
               ("answer", "citations", "reasoning", "input_tokens", "output_tokens", "latency")},
        })
        return result, chunks

    for i, record in enumerate(sample, 1):
        qid = record["financebench_id"]
        hit = cached.get(qid)
        usable = hit is not None and hit.get("answer") != "ERROR"
        # A cached refusal from a smaller top_k still deserves the wider retry.
        retry_due = (usable and retry_top_k > hit.get("top_k", top_k)
                     and is_refusal(hit["answer"]))

        try:
            if usable and not retry_due:
                result = hit
                chunks = []  # not re-retrieved; retrieval detail came from the cached run
            else:
                if retry_due:
                    result, chunks = None, []
                else:
                    result, chunks = ask(record, top_k)
                # Refusal usually means the needed figures were not in the top
                # few chunks (measured: 31/61 refusals at top_k=5), so retry once
                # with a wider window before accepting NOT FOUND.
                if result is None or (is_refusal(result["answer"]) and retry_top_k > top_k):
                    escalated += 1
                    result, chunks = ask(record, retry_top_k)
                    if not is_refusal(result["answer"]):
                        rescued += 1
        except DailyLimitReached as exc:
            print(f"\n{exc}. {i - 1} of {len(sample)} questions processed; "
                  "rerun the same command later to continue.", flush=True)
            stopped_early = True
            break

        verdict, note = grade(str(record["answer"]), result["answer"])
        verdicts.append(verdict)
        by_type_verdicts[record["question_type"]].append(verdict)
        total_in += result.get("input_tokens", 0)
        total_out += result.get("output_tokens", 0)

        retrieved = (
            sorted({c["doc_name"] for c in chunks}) if chunks
            else result.get("retrieved_docs", [])
        )
        # flush=True so an interrupted run still shows what it completed
        print("=" * 78, flush=True)
        print(f"Q{i} [{record['question_type']}]  {qid}")
        print(f"  QUESTION : {safe(record['question'])[:300]}")
        print(f"  EXPECTED : {safe(str(record['answer']))[:300]}")
        print(f"  GENERATED: {safe(result['answer'])[:300]}")
        print(f"  CITED    : {safe(result.get('citations', ''))[:120]}")
        print(f"  RETRIEVED: {', '.join(retrieved)}")
        print(f"  TRUTH DOC: {record['doc_name']}")
        print(f"  VERDICT  : {verdict}" + (f"  ({note})" if note else ""))
        print(f"  tokens in/out: {result.get('input_tokens',0)}/{result.get('output_tokens',0)}"
              f"  latency {result.get('latency',0):.1f}s", flush=True)

    retriever.close()

    print("\n" + "=" * 78)
    print("SUMMARY" + (" (PARTIAL -- daily quota reached)" if stopped_early else ""))
    print("=" * 78)
    counts = {v: verdicts.count(v) for v in ("CORRECT", "WRONG", "REFUSED", "REVIEW", "ERROR")}
    total = len(verdicts)
    if not total:
        print("  No questions answered yet.")
        return
    for verdict, count in counts.items():
        if count:
            print(f"  {verdict:8s} {count:2d}/{total}  ({count/total:.0%})")

    auto_graded = counts["CORRECT"] + counts["WRONG"]
    if auto_graded:
        print(f"\n  Accuracy on numerically-graded answers: "
              f"{counts['CORRECT']}/{auto_graded}  ({counts['CORRECT']/auto_graded:.1%})")
    if counts["REVIEW"]:
        print(f"  {counts['REVIEW']} free-text answer(s) need your judgement (marked REVIEW)")
    if escalated:
        print(f"  Fallback: {escalated} NOT FOUND answer(s) at top_k={top_k} retried with "
              f"top_k={retry_top_k}; {rescued} produced an answer")

    print("\n  --- by question type ---")
    for qtype in sorted(by_type_verdicts):
        vs = by_type_verdicts[qtype]
        correct = vs.count("CORRECT")
        graded = correct + vs.count("WRONG")
        rate = f"{correct}/{graded} ({correct/graded:.0%})" if graded else "n/a"
        print(f"  {qtype:20s} n={len(vs):2d}  auto-graded {rate:14s} "
              f"refused={vs.count('REFUSED')} review={vs.count('REVIEW')} err={vs.count('ERROR')}")

    print(f"\n  Tokens: {total_in} in / {total_out} out  "
          f"(~{(total_in + total_out) / total:.0f} per question)")


if __name__ == "__main__":
    main()
