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
REPORT_PATH = ROOT / "documentation" / "reports" / "retrieval_report.pdf"

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


def page_is_evidence(hit: dict, evidence: list[dict]) -> bool:
    """A chunk counts as a hit if it spans the labelled evidence page of that document."""
    for ev in evidence:
        if ev["doc_name"] != hit["doc_name"]:
            continue
        if hit["page_start"] <= ev["evidence_page_num"] <= hit["page_end"]:
            return True
    return False


def evaluate(retriever: Retriever) -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    results = []
    start = time.time()
    for i, record in enumerate(questions, 1):
        hits = retriever.search(record["question"], top_k=TOP_K)
        answer = str(record["answer"])

        scored_hits = []
        for hit in hits:
            scored_hits.append(
                {
                    **hit,
                    "page_match": page_is_evidence(hit, record["evidence"]),
                    "answer_match": answer_in_text(answer, hit["text"]),
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
            }
        )

        if i % 25 == 0:
            print(f"  {i}/{len(questions)} questions | {time.time() - start:.0f}s")

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    hit1 = sum(1 for r in results if r["first_page_rank"] == 1)
    hit5 = sum(1 for r in results if r["page_hit"])
    mrr = sum(1 / r["first_page_rank"] for r in results if r["first_page_rank"]) / n
    answer_hit = sum(1 for r in results if r["answer_hit"])
    return {
        "total": n,
        "hit1": hit1,
        "hit5": hit5,
        "mrr": mrr,
        "answer_hit": answer_hit,
    }


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _verdict_cell(passed: bool) -> str:
    return "HIT" if passed else "MISS"


def build_pdf(results: list[dict], stats: dict):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=8, leading=10)
    small = ParagraphStyle("small", parent=body, fontSize=7, leading=8.5,
                           textColor=colors.HexColor("#444444"))
    qstyle = ParagraphStyle("q", parent=styles["Heading4"], fontSize=9.5, leading=12,
                            spaceAfter=2)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4,
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="FinanceBench Retrieval Report",
    )

    story = [
        Paragraph("FinanceBench Retrieval Evaluation", styles["Title"]),
        Paragraph(
            "Hybrid retrieval (BM25 + dense embeddings, RRF fusion) &mdash; "
            f"top {TOP_K} results per question. No re-ranking, no metadata filtering.",
            body,
        ),
        Spacer(1, 6 * mm),
    ]

    n = stats["total"]
    summary_rows = [
        ["Metric", "Result", "What it means"],
        ["Hit@1", f"{stats['hit1']}/{n}  ({stats['hit1']/n:.1%})",
         "Correct evidence page was the top result"],
        ["Hit@5", f"{stats['hit5']}/{n}  ({stats['hit5']/n:.1%})",
         f"Correct evidence page appeared in top {TOP_K}"],
        ["MRR", f"{stats['mrr']:.3f}",
         "Average of 1/(rank of first correct page)"],
        ["Answer in text", f"{stats['answer_hit']}/{n}  ({stats['answer_hit']/n:.1%})",
         "Expected answer value found in retrieved text"],
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
        overall = "PAGE HIT" if r["page_hit"] else "PAGE MISS"
        overall_color = "#1a7f37" if r["page_hit"] else "#c1121f"

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

        rows = [["#", "Document", "Pages", "Page", "Answer", "Retrieved text (preview)"]]
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
                h["doc_name"][:26],
                f"{h['page_start']}-{h['page_end']}",
                _verdict_cell(h["page_match"]),
                _verdict_cell(h["answer_match"]),
                Paragraph(_esc(preview), small),
            ])
            row_i = len(rows) - 1
            for col, passed in ((3, h["page_match"]), (4, h["answer_match"])):
                style_cmds.append((
                    "BACKGROUND", (col, row_i), (col, row_i),
                    colors.HexColor("#c6f0c8") if passed else colors.HexColor("#f8cdcd"),
                ))

        table = Table(rows, colWidths=[6 * mm, 40 * mm, 14 * mm, 13 * mm, 15 * mm, 80 * mm])
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
    retriever = Retriever(use_reranker=False)
    print(f"\nEvaluating {TOP_K}-result retrieval on all questions ...")
    results = evaluate(retriever)
    retriever.close()

    stats = summarize(results)
    n = stats["total"]
    print("\n===== RETRIEVAL BASELINE =====")
    print(f"Questions:       {n}")
    print(f"Hit@1:           {stats['hit1']}/{n}  ({stats['hit1']/n:.1%})")
    print(f"Hit@5:           {stats['hit5']}/{n}  ({stats['hit5']/n:.1%})")
    print(f"MRR:             {stats['mrr']:.3f}")
    print(f"Answer in text:  {stats['answer_hit']}/{n}  ({stats['answer_hit']/n:.1%})")

    build_pdf(results, stats)
    print(f"\nPDF report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
