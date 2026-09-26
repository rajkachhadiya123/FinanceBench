"""
Validation: how well does raw PDF text extraction (PyMuPDF) capture the
content FinanceBench's evidence_text points to?

Key insight (found via manual diffing): evidence_text is frequently a SHORT
EXCERPT from within a page, not a full-page dump. So the right check is
"is (a near-match of) evidence_text CONTAINED in the extracted page text?",
not "does extracted page text equal evidence_text?".
"""
import json
import difflib
from pathlib import Path
from collections import defaultdict

import fitz  # pymupdf

ROOT = Path(__file__).resolve().parents[1]
JSONL_PATH = ROOT / "data" / "EVAL" / "financebench_open_source.jsonl"
PDF_DIR = ROOT / "data" / "RAW"


def normalize(text: str) -> str:
    return " ".join(text.split())


def safe(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def containment_ratio(gt: str, extracted: str) -> float:
    """
    What fraction of the (normalized) ground-truth text's characters are
    covered by contiguous matching blocks found in the extracted text.
    1.0 = GT is fully found inside extracted (exact or near-exact substring).
    """
    if not gt:
        return 1.0
    sm = difflib.SequenceMatcher(None, gt, extracted, autojunk=False)
    matched_chars = sum(block.size for block in sm.get_matching_blocks())
    return matched_chars / len(gt)


def load_evidence_samples():
    with open(JSONL_PATH, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    samples = []
    for rec in records:
        for ev in rec["evidence"]:
            samples.append(
                {
                    "financebench_id": rec["financebench_id"],
                    "doc_name": ev["doc_name"],
                    "page_num": ev["evidence_page_num"],
                    "gt_text": ev["evidence_text"],
                }
            )
    return samples


def main():
    samples = load_evidence_samples()
    print(f"Total evidence entries: {len(samples)}")

    doc_cache = {}
    scores = []
    failures = []
    missing_pdfs = set()
    by_doc_type_scores = defaultdict(list)

    for s in samples:
        doc_name = s["doc_name"]
        pdf_path = PDF_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            missing_pdfs.add(doc_name)
            continue

        if doc_name not in doc_cache:
            doc_cache[doc_name] = fitz.open(pdf_path)
        doc = doc_cache[doc_name]

        page_num = s["page_num"]
        if page_num < 0 or page_num >= doc.page_count:
            failures.append((s, "PAGE_OUT_OF_RANGE", None))
            continue

        extracted = normalize(doc[page_num].get_text())
        gt = normalize(s["gt_text"])

        score = containment_ratio(gt, extracted)
        scores.append(score)

        # bucket by doc "type" suffix for pattern-spotting
        doc_type = doc_name.split("_")[-1]
        by_doc_type_scores[doc_type].append(score)

        if score < 0.90:
            failures.append((s, f"LOW_CONTAINMENT={score:.2f}", extracted[:250]))

    for doc in doc_cache.values():
        doc.close()

    print(f"\nEvaluated: {len(scores)} evidence entries across {len(doc_cache)} PDFs")
    if scores:
        print(f"Average containment: {sum(scores)/len(scores):.3f}")
        print(f"Min containment: {min(scores):.3f}")
        print(f"Entries below 0.90 containment: {sum(1 for x in scores if x < 0.90)} / {len(scores)}")
        print(f"Entries below 0.50 containment (likely real failures): {sum(1 for x in scores if x < 0.50)} / {len(scores)}")

    print("\n=== Average containment by document-type suffix ===")
    for doc_type, sc in sorted(by_doc_type_scores.items(), key=lambda x: sum(x[1]) / len(x[1])):
        print(f"  {doc_type:20s}: avg={sum(sc)/len(sc):.3f}  n={len(sc)}")

    if missing_pdfs:
        print(f"\nMissing PDFs ({len(missing_pdfs)}): {sorted(missing_pdfs)[:10]}")

    real_failures = [f for f in failures if f[1].startswith("LOW_CONTAINMENT") and float(f[1].split("=")[1]) < 0.5]
    real_failures += [f for f in failures if f[1] == "PAGE_OUT_OF_RANGE"]
    print(f"\n=== {len(real_failures)} LIKELY REAL FAILURES (containment < 0.50) ===")
    for s, reason, snippet in real_failures[:15]:
        print(f"\n- {s['doc_name']} page {s['page_num']} | {reason}")
        print(f"  GT   : {safe(normalize(s['gt_text']))[:200]}")
        if snippet:
            print(f"  Extr : {safe(snippet)[:200]}")


if __name__ == "__main__":
    main()
