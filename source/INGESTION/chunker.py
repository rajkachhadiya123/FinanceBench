"""
Fixed-size sliding-window chunker for FinanceBench PDFs.

Extracts each PDF's pages as text and concatenates them into one continuous
stream (so a topic that spans a page boundary isn't cut at that boundary),
then cuts it into chunks of a fixed size (2000 characters, ~500 tokens) with
a fixed overlap (300 characters, ~75 tokens) between consecutive chunks.
The step between chunk starts is always the same (chunk_size - overlap).

Cuts happen at an exact character count, with no word-boundary snapping --
a word or number can end up split across the chunk boundary (e.g. "$5,363"
-> "$5,3" | "63"), but the 300-character overlap guarantees the complete,
unsplit version always appears in full in the following chunk, so nothing
is ever only available in split form.

Each chunk keeps track of which page(s) of the source PDF it came from,
so retrieval results can still be scored against FinanceBench's
page-indexed ground truth.
"""
import json
from pathlib import Path

import fitz  # pymupdf

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "RAW"
OUTPUT_PATH = ROOT / "data" / "PROCESSED" / "chunks.jsonl"

CHUNK_SIZE = 2000  # characters, ~500 tokens
CHUNK_OVERLAP = 300  # characters, ~75 tokens
STEP = CHUNK_SIZE - CHUNK_OVERLAP
MIN_CHUNK_LENGTH = 300  # merge a trailing chunk shorter than this into the previous one


def extract_pages(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    pages = [doc[i].get_text() for i in range(doc.page_count)]
    doc.close()
    return pages


def build_full_text_with_page_offsets(pages: list[str]):
    """Concatenate pages; record the character offset each page starts at."""
    full_text_parts = []
    page_offsets = []  # page_offsets[i] = start offset of page i in full_text
    offset = 0
    for page_text in pages:
        page_offsets.append(offset)
        full_text_parts.append(page_text)
        offset += len(page_text)
        full_text_parts.append("\n\n")
        offset += 2
    return "".join(full_text_parts), page_offsets


def offset_to_page(offset: int, page_offsets: list[int]) -> int:
    """Which page index does this character offset fall in?"""
    lo, hi = 0, len(page_offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if page_offsets[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sliding_window_chunks(text: str, chunk_size: int, step: int):
    """Fixed-step sliding window over `text`, returning (start, end) spans."""
    spans = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        if end > i:
            spans.append((i, end))
        if end >= n:
            break
        i += step

    # A document's last chunk can end up as just a few leftover words; fold
    # it into the previous chunk instead of keeping it as a near-empty entry.
    if len(spans) >= 2:
        last_start, last_end = spans[-1]
        if (last_end - last_start) < MIN_CHUNK_LENGTH:
            prev_start, _ = spans[-2]
            spans[-2] = (prev_start, last_end)
            spans.pop()

    return spans


def chunk_document(pdf_path: Path) -> list[dict]:
    doc_name = pdf_path.stem
    pages = extract_pages(pdf_path)
    full_text, page_offsets = build_full_text_with_page_offsets(pages)

    chunk_spans = sliding_window_chunks(full_text, CHUNK_SIZE, STEP)

    chunks = []
    for idx, (start, end) in enumerate(chunk_spans):
        text = full_text[start:end].strip()
        if not text:
            continue
        page_start = offset_to_page(start, page_offsets)
        page_end = offset_to_page(max(end - 1, start), page_offsets)
        chunks.append(
            {
                "chunk_id": f"{doc_name}__{idx:04d}",
                "doc_name": doc_name,
                "page_start": page_start,
                "page_end": page_end,
                "char_start": start,
                "char_end": end,
                "text": text,
            }
        )
    return chunks


def main():
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs to chunk.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    chunk_lengths = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, pdf_path in enumerate(pdf_paths, 1):
            chunks = chunk_document(pdf_path)
            for c in chunks:
                out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                chunk_lengths.append(len(c["text"]))
            total_chunks += len(chunks)
            if i % 50 == 0 or i == len(pdf_paths):
                print(f"  processed {i}/{len(pdf_paths)} PDFs, {total_chunks} chunks so far")

    print(f"\nDone. Total chunks: {total_chunks}")
    if chunk_lengths:
        print(f"Chunk length (chars): min={min(chunk_lengths)}, "
              f"avg={sum(chunk_lengths)/len(chunk_lengths):.0f}, max={max(chunk_lengths)}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
