"""
Runs all 150 FinanceBench questions through the retriever (top 5) and writes
a PDF report marking, per question, whether retrieval found the right page and
whether the expected answer's value actually appears in the retrieved text.

Two independent checks, because they fail for different reasons:
  PAGE   -- did we retrieve the page FinanceBench labels as the evidence?
  ANSWER -- does the retrieved text actually contain the expected answer?

A question can pass PAGE but fail ANSWER (right page, but the chunk covering
it cut off the figure), or fail PAGE but pass ANSWER (found the number
somewhere else -- often a different fiscal year's filing repeating it, which
is a false positive worth seeing).
"""
import json
import re
import sys
import time
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.RETRIEVAL.retriever import Retriever  # noqa: E402

QUESTIONS_PATH = ROOT / "data" / "EVAL" / "financebench_open_source.jsonl"
REPORTS_DIR = ROOT / "documentation" / "reports"

TOP_K = 5
NUMBER_RE = re.compile(r"\(?\$?\s?-?\d[\d,]*\.?\d*\)?%?")
WORD_RE = re.compile(r"[a-z]{4,}")
KEYWORD_PASS_RATIO = 0.5

FILLER_WORDS = {
    "that", "this", "with", "from", "have", "been", "were", "which", "their",
    "there", "these", "those", "would", "could", "should", "about", "than",
    "then", "they", "them", "when", "what", "will", "year", "years", "fiscal",
    "also", "into", "more", "most", "such", "some", "because", "company",
}


def extract_numbers(text: str) -> set[float]:
    """Pull numeric values out of text, normalizing $, commas, %, and (negatives)."""
    values = set()
    for raw in NUMBER_RE.findall(text):
        cleaned = raw.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        values.add(-value if negative else value)
    return values


def number_present(target: float, candidates: set[float], rel_tol: float = 0.01) -> bool:
    """
    True if `target` appears among `candidates`, allowing 1% rounding slack and
    unit-scale differences (an answer in billions vs a filing reporting millions).
    """
    scales = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
    for candidate in candidates:
        for scale in scales:
            scaled = target * scale
            if abs(scaled - candidate) <= rel_tol * max(abs(scaled), abs(candidate), 1e-9):
                return True
    return False


def keyword_ratio(answer: str, text: str) -> float:
    """Fraction of the answer's content words that appear in `text`."""
    words = {w for w in WORD_RE.findall(answer.lower()) if w not in FILLER_WORDS}
    if not words:
        return 0.0
    lowered = text.lower()
    return sum(1 for w in words if w in lowered) / len(words)


def answer_in_text(answer: str, text: str) -> bool:
    answer_numbers = extract_numbers(answer)
    if answer_numbers:
        text_numbers = extract_numbers(text)
        return any(number_present(n, text_numbers) for n in answer_numbers)
    return keyword_ratio(answer, text) >= KEYWORD_PASS_RATIO


def answer_is_retrievable(record: dict) -> bool:
    """
    Could ANY retriever satisfy this question's answer check?

    38% of FinanceBench answers must be computed (ratios, multi-year averages,
    year-over-year deltas), so the expected value appears nowhere in the
    filing. Scoring those as retrieval failures understates quality by ~17pp,
    so they are excluded from the "% of achievable" denominator.
    """
    evidence_text = "\n".join(ev["evidence_text"] for ev in record["evidence"])
    return answer_in_text(str(record["answer"]), evidence_text)


def page_is_evidence(hit: dict, evidence: list[dict]) -> bool:
    """A chunk counts as a hit if it spans the labelled evidence page of that document."""
    for ev in evidence:
        if ev["doc_name"] != hit["doc_name"]:
            continue
        if hit["page_start"] <= ev["evidence_page_num"] <= hit["page_end"]:
            return True
    return False


def evaluate(retriever: Retriever, config: dict) -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    results = []
    start = time.time()
    for i, record in enumerate(questions, 1):
        hits = retriever.search(
            record["question"],
            top_k=TOP_K,
            metadata_filter=config["filter"],
            dedup=config["dedup"],
            rerank=config["rerank"],
            oracle_doc=record["doc_name"] if config["oracle"] else None,
        )
        answer = str(record["answer"])
        expected_docs = {ev["doc_name"] for ev in record["evidence"]}

        scored_hits = []
        for hit in hits:
            right_doc = hit["doc_name"] in expected_docs
            has_answer = answer_in_text(answer, hit["text"])
            scored_hits.append(
                {
                    **hit,
                    "page_match": page_is_evidence(hit, record["evidence"]),
                    "doc_match": right_doc,
                    "answer_match": has_answer,
                    # The metric that actually matters: correct value AND correct
                    # source. Plain answer_match rewards finding a 2018 figure in
                    # the 2021 filing, which is a wrong answer with a right number.
                    "grounded_answer": right_doc and has_answer,
                }
            )

        page_ranks = [h["rank"] for h in scored_hits if h["page_match"]]
        results.append(
            {
                "financebench_id": record["financebench_id"],
                "question": record["question"],
                "answer": answer,
                "expected": sorted(
                    {(ev["doc_name"], ev["evidence_page_num"]) for ev in record["evidence"]}
                ),
                "hits": scored_hits,
                "page_hit": bool(page_ranks),
                "first_page_rank": min(page_ranks) if page_ranks else None,
                "answer_hit": any(h["answer_match"] for h in scored_hits),
                "doc_hit": any(h["doc_match"] for h in scored_hits),
                "grounded_hit": any(h["grounded_answer"] for h in scored_hits),
                "question_type": record["question_type"],
                "retrievable": answer_is_retrievable(record),
            }
        )

        if i % 25 == 0:
            print(f"  {i}/{len(questions)} questions | {time.time() - start:.0f}s")

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    ranks = [r["first_page_rank"] for r in results]
    return {
        "total": n,
        "hit1": sum(1 for r in ranks if r == 1),
        "hit3": sum(1 for r in ranks if r is not None and r <= 3),
        "hit5": sum(1 for r in results if r["page_hit"]),
        "mrr": sum(1 / r for r in ranks if r) / n,
        "answer_hit": sum(1 for r in results if r["answer_hit"]),
        "doc_hit": sum(1 for r in results if r["doc_hit"]),
        "grounded_hit": sum(1 for r in results if r["grounded_hit"]),
        # Three-way breakdown: every question lands in exactly one bucket.
        "right_doc_right_page": sum(1 for r in results if r["page_hit"]),
        "right_doc_wrong_page": sum(
            1 for r in results if r["doc_hit"] and not r["page_hit"]
        ),
        "wrong_doc": sum(1 for r in results if not r["doc_hit"]),
        # Questions whose answer is written down somewhere and could therefore
        # be retrieved at all -- the honest denominator.
        "retrievable": sum(1 for r in results if r["retrievable"]),
        "grounded_of_retrievable": sum(
            1 for r in results if r["retrievable"] and r["grounded_hit"]
        ),
    }


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _verdict_cell(passed: bool) -> str:
    return "HIT" if passed else "MISS"


def build_pdf(results: list[dict], stats: dict, config: dict, report_path: Path):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8, leading=10)
    small = ParagraphStyle("small", parent=body, fontSize=7, leading=8.5,
                           textColor=colors.HexColor("#444444"))
    qstyle = ParagraphStyle("q", parent=styles["Heading4"], fontSize=9.5, leading=12,
                            spaceAfter=2)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(report_path), pagesize=A4,
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="FinanceBench Retrieval Report",
    )

    enabled = [name for name in ("filter", "dedup", "rerank", "oracle") if config[name]]
    config_line = ", ".join(enabled) if enabled else "none (baseline)"

    story = [
        Paragraph("FinanceBench Retrieval Evaluation", styles["Title"]),
        Paragraph(
            "Hybrid retrieval (BM25 + dense embeddings, RRF fusion) &mdash; "
            f"top {TOP_K} results per question.<br/>"
            f"<b>Stages enabled:</b> {config_line}",
            body,
        ),
    ]
    if config["oracle"]:
        story.append(Paragraph(
            '<font color="#c1121f"><b>DIAGNOSTIC ONLY &mdash; NOT A REAL RESULT.</b></font> '
            "This run filters using the benchmark's ground-truth document name, "
            "which a deployed system would not have. It measures the ceiling that "
            "perfect company/year extraction could reach, nothing more.",
            body,
        ))
    story.append(Spacer(1, 6 * mm))

    n = stats["total"]
    summary_rows = [
        ["Metric", "Result", "What it means"],
        ["Grounded answer", f"{stats['grounded_hit']}/{n}  ({stats['grounded_hit']/n:.1%})",
         "PRIMARY: expected value found, in the correct document"],
        ["...of achievable",
         f"{stats['grounded_of_retrievable']}/{stats['retrievable']}  "
         f"({stats['grounded_of_retrievable']/stats['retrievable']:.1%})",
         f"Excludes {n - stats['retrievable']} questions whose answer must be "
         "computed (ratios, averages, deltas) and appears in no filing"],
        ["Correct document", f"{stats['doc_hit']}/{n}  ({stats['doc_hit']/n:.1%})",
         f"Correct filing appeared in top {TOP_K} (any page)"],
        ["Hit@1 (page)", f"{stats['hit1']}/{n}  ({stats['hit1']/n:.1%})",
         "Labelled evidence page was the top result"],
        ["Hit@3 (page)", f"{stats['hit3']}/{n}  ({stats['hit3']/n:.1%})",
         "Labelled evidence page in top 3"],
        ["Hit@5 (page)", f"{stats['hit5']}/{n}  ({stats['hit5']/n:.1%})",
         "Labelled evidence page in top 5 (strict: ignores other valid pages)"],
        ["MRR", f"{stats['mrr']:.3f}",
         "Average of 1/(rank of first correct page)"],
        ["Answer in text", f"{stats['answer_hit']}/{n}  ({stats['answer_hit']/n:.1%})",
         "Loose: value found anywhere, even in the wrong year's filing"],
        ["", "", ""],
        ["Right doc + page", f"{stats['right_doc_right_page']}/{n} "
                            f" ({stats['right_doc_right_page']/n:.1%})", "Breakdown bucket 1"],
        ["Right doc, wrong page", f"{stats['right_doc_wrong_page']}/{n} "
                                 f" ({stats['right_doc_wrong_page']/n:.1%})", "Breakdown bucket 2"],
        ["Wrong document", f"{stats['wrong_doc']}/{n}  ({stats['wrong_doc']/n:.1%})",
         "Breakdown bucket 3"],
    ]
    summary = Table(summary_rows, colWidths=[30 * mm, 38 * mm, 110 * mm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 1), (1, -1), "Helvetica-Bold"),
    ]))
    story += [summary, PageBreak()]

    for idx, r in enumerate(results, 1):
        overall = "GROUNDED" if r["grounded_hit"] else (
            "DOC ONLY" if r["doc_hit"] else "MISS"
        )
        overall_color = "#1a7f37" if r["grounded_hit"] else (
            "#b36b00" if r["doc_hit"] else "#c1121f"
        )

        expected = ", ".join(f"{d} p.{p}" for d, p in r["expected"])
        block = [
            Paragraph(f"Q{idx}. {_esc(r['question'])}", qstyle),
            Paragraph(
                f"<b>Expected answer:</b> {_esc(r['answer'][:300])}<br/>"
                f"<b>Evidence:</b> {_esc(expected)}",
                small,
            ),
            Spacer(1, 1.5 * mm),
        ]

        rows = [["#", "Document", "Pages", "Doc", "Page", "Ans", "Retrieved text (preview)"]]
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("FONTSIZE", (0, 0), (-1, -1), 6.5),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for h in r["hits"]:
            preview = " ".join(h["text"].split())[:150]
            rows.append([
                str(h["rank"]),
                h["doc_name"][:24],
                f"{h['page_start']}-{h['page_end']}",
                _verdict_cell(h["doc_match"]),
                _verdict_cell(h["page_match"]),
                _verdict_cell(h["answer_match"]),
                Paragraph(_esc(preview), small),
            ])
            row_i = len(rows) - 1
            for col, passed in (
                (3, h["doc_match"]), (4, h["page_match"]), (5, h["answer_match"])
            ):
                style_cmds.append((
                    "BACKGROUND", (col, row_i), (col, row_i),
                    colors.HexColor("#c6f0c8") if passed else colors.HexColor("#f8cdcd"),
                ))

        table = Table(
            rows, colWidths=[6 * mm, 36 * mm, 13 * mm, 11 * mm, 11 * mm, 11 * mm, 74 * mm]
        )
        table.setStyle(TableStyle(style_cmds))

        block += [
            table,
            Paragraph(f'<font color="{overall_color}"><b>{overall}</b></font>'
                      + (f' (rank {r["first_page_rank"]})' if r["first_page_rank"] else ""), small),
            Spacer(1, 5 * mm),
        ]
        story.append(KeepTogether(block))

    doc.build(story)


def main():
    config = {
        "filter": "--filter" in sys.argv,
        "dedup": "--dedup" in sys.argv,
        "rerank": "--rerank" in sys.argv,
        "oracle": "--oracle" in sys.argv,
    }
    enabled = [name for name, on in config.items() if on]
    label = "+".join(enabled) if enabled else "baseline"

    if config["oracle"]:
        print("\n" + "!" * 72)
        print("! ORACLE MODE -- DIAGNOSTIC ONLY, NOT A REAL RESULT.")
        print("! Filters using the benchmark's ground-truth document name, which a")
        print("! deployed system would never have. Measures a ceiling, not accuracy.")
        print("!" * 72)

    retriever = Retriever(use_reranker=config["rerank"])
    print(f"\nEvaluating top-{TOP_K} retrieval -- config: {label} ...")
    start = time.time()
    results = evaluate(retriever, config)
    elapsed = time.time() - start
    retriever.close()

    stats = summarize(results)
    n = stats["total"]
    print(f"\n===== RETRIEVAL: {label} =====")
    print(f"Questions:              {n}     ({elapsed:.0f}s, {elapsed/n:.2f}s/question)")
    retrievable = stats["retrievable"]
    print(f"GROUNDED ANSWER:        {stats['grounded_hit']}/{n}  "
          f"({stats['grounded_hit']/n:.1%})   <-- primary metric")
    print(f"  of ACHIEVABLE:        {stats['grounded_of_retrievable']}/{retrievable}  "
          f"({stats['grounded_of_retrievable']/retrievable:.1%})   <-- excludes the "
          f"{n - retrievable} questions whose answer must be computed")
    print(f"Correct document:       {stats['doc_hit']}/{n}  ({stats['doc_hit']/n:.1%})")
    print(f"Hit@1 (page):           {stats['hit1']}/{n}  ({stats['hit1']/n:.1%})")
    print(f"Hit@3 (page):           {stats['hit3']}/{n}  ({stats['hit3']/n:.1%})")
    print(f"Hit@5 (page):           {stats['hit5']}/{n}  ({stats['hit5']/n:.1%})")
    print(f"MRR:                    {stats['mrr']:.3f}")
    print(f"Answer in text (loose): {stats['answer_hit']}/{n}  ({stats['answer_hit']/n:.1%})")
    print("\n--- breakdown (each question in exactly one bucket) ---")
    print(f"Right doc AND right page: {stats['right_doc_right_page']}/{n}  "
          f"({stats['right_doc_right_page']/n:.1%})")
    print(f"Right doc, WRONG page:    {stats['right_doc_wrong_page']}/{n}  "
          f"({stats['right_doc_wrong_page']/n:.1%})")
    print(f"WRONG document:           {stats['wrong_doc']}/{n}  "
          f"({stats['wrong_doc']/n:.1%})")

    report_path = REPORTS_DIR / f"retrieval_report_{label.replace('+', '_')}.pdf"
    build_pdf(results, stats, config, report_path)
    print(f"\nPDF report written to: {report_path}")


if __name__ == "__main__":
    main()
