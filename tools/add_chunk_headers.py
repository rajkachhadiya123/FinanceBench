"""
Prefix every existing chunk with a context header (V3 step 3, "metadata in the chunk").

    [VERIZON | 10-K | FY2022 | page 80 | Item 15 > Schedule I - Parent Company Only]
    Total current liabilities 6,831 7,224 ...

Why: chunks carried no identity, so retrieval, the reranker and the LLM could
not tell e.g. a parent-company-only schedule from the consolidated balance
sheet (5 of 16 wrong answers in the v2 analysis). A prompt rule alone could
not fix it (prompt v3: 0 of 16 fixed) because the heading was not in the chunk.

How: each filing is read top to bottom exactly as chunker.py concatenated it,
tracking the current "Item" (10-K/10-Q structure) and "Section" (financial
statement, notes, individual note, parent-company schedule). Headings are
short standalone lines; the same words inside a sentence are ignored, as are
table-of-contents pages. Each chunk gets the state in force at its char_start,
so a statement that runs over several pages keeps its label (carry-forward).

Chunk boundaries, order and count are unchanged, so chunk_meta.jsonl and the
retriever's position contract still hold.

Outputs (data/PROCESSED/):
    chunks_headed.jsonl   same chunks, text = header + "\\n" + original text
    chunks_headed.zip     the above, for the Colab embedding run

Usage:
    python tools/add_chunk_headers.py            # build + print a review sample
    python tools/add_chunk_headers.py --sample   # review sample only, writes nothing
"""
import bisect
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.INGESTION.chunker import (  # noqa: E402  (same text the chunks were cut from)
    PDF_DIR,
    build_full_text_with_page_offsets,
    extract_pages,
)
from source.RETRIEVAL.metadata_filter import YEAR_IN_NAME_RE, ANY_YEAR_RE  # noqa: E402

PROCESSED = ROOT / "data" / "PROCESSED"
CHUNKS_IN = PROCESSED / "chunks.jsonl"
CHUNKS_OUT = PROCESSED / "chunks_headed.jsonl"
ZIP_OUT = PROCESSED / "chunks_headed.zip"

MAX_HEADING_LEN = 90    # a heading is a short standalone line
MAX_ITEM_LEN = 160      # Item titles can be long ("Item 7. Management's Discussion and Analysis of ...")
TOC_ITEMS_PER_PAGE = 3  # a page listing this many distinct Items is a table of contents

ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?(?:\.\d{2})?)\s*[.:\-–—]?\s*(.*)$", re.I)
STATEMENT_RE = re.compile(
    r"^(condensed\s+)?(consolidated\s+)?(balance\s+sheets?|statements?\s+of\s+[a-z ,'’()&-]{3,60})$",
    re.I)
NOTES_RE = re.compile(r"^notes?\s+to\s+(the\s+)?(condensed\s+)?(consolidated\s+)?financial\s+statements", re.I)
NOTE_N_RE = re.compile(r"^note\s+(\d{1,2})\s*[.:\-–—]?\s*([A-Za-z].{2,70})?$", re.I)
# Only the schedule's actual title wording. A bare "Schedule I" also appears in
# indentures, debt tables and exhibit lists, and wrongly labelled whole notes.
PARENT_RE = re.compile(
    r"condensed\s+financial\s+information\s+of\s+(the\s+)?registrant|parent\s+company\s+only", re.I)


def clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_sentence(line: str) -> bool:
    """Running text, not a heading: ends like a sentence or reads like one."""
    return line.endswith((".", ",", ";")) and len(line) > 40 or \
        bool(re.search(r"\b(see|refer|included|following|described|presented|our)\b", line, re.I))


def doc_label(doc_name: str) -> str:
    """'JOHNSON_JOHNSON_2022Q4_EARNINGS' -> 'JOHNSON JOHNSON | EARNINGS | FY2022 Q4'."""
    parts = doc_name.split("_")
    for i, part in enumerate(parts):
        m = YEAR_IN_NAME_RE.match(part)
        if m:
            company = " ".join(parts[:i]).upper()
            doc_type = " ".join(parts[i + 1:]).upper().replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
            period = f"FY{m.group(1)}" + (f" {m.group(2)}" if m.group(2) else "")
            return f"{company} | {doc_type or 'FILING'} | {period}"
    year = ANY_YEAR_RE.search(doc_name)
    return f"{parts[0].upper()} | {' '.join(parts[1:]).upper()} | {'FY' + year.group() if year else ''}".rstrip(" |")


def heading_events(full_text: str, page_offsets: list[int]) -> list[tuple[int, str, str]]:
    """(char offset, 'item' | 'section', label) for every heading, in document order."""
    lines, offset = [], 0
    for raw in full_text.split("\n"):
        lines.append((offset, clean(raw)))
        offset += len(raw) + 1

    def page_of(off):
        return bisect.bisect_right(page_offsets, off) - 1

    def next_line(idx):
        return next((l for _, l in lines[idx + 1:idx + 4] if l), "")

    # Table-of-contents pages list many Items; their headings would set a wrong state.
    items_per_page: dict[int, set] = {}
    for off, l in lines:
        m = ITEM_RE.match(l) if l and len(l) <= MAX_ITEM_LEN else None
        if m:
            items_per_page.setdefault(page_of(off), set()).add(m.group(1).upper())
    toc_pages = {p for p, items in items_per_page.items() if len(items) >= TOC_ITEMS_PER_PAGE}

    events = []
    for idx, (off, line) in enumerate(lines):
        if not line or page_of(off) in toc_pages:
            continue
        m = ITEM_RE.match(line)
        if m and len(line) <= MAX_ITEM_LEN and not is_sentence(line):
            title = m.group(2).strip(" .") or next_line(idx)  # title may be on the next line
            events.append((off, "item", f"Item {m.group(1).upper()} {title[:60]}".strip()))
            continue
        if len(line) > MAX_HEADING_LEN or is_sentence(line):
            continue
        if PARENT_RE.search(line):
            if re.search(r"registrant|^schedule", line, re.I):  # the schedule's own title
                events.append((off, "section", "Schedule I - Parent Company Only (Condensed Financial Information of Registrant)"))
            else:  # a caption like "(Parent Company only)" on a table inside a note
                events.append((off, "sub", "Parent Company Only"))
        elif NOTES_RE.match(line):
            events.append((off, "section", "Notes to Financial Statements"))
        elif NOTE_N_RE.match(line):
            m = NOTE_N_RE.match(line)
            title = (m.group(2) or next_line(idx)).strip(" .")[:60]
            events.append((off, "section", f"Note {m.group(1)} {title}".strip()))
        elif STATEMENT_RE.match(line) and ("consolidated" in line.lower() or "balance sheet" in line.lower()):
            events.append((off, "section", line.title()[:80]))
    return events


def build_headers(doc_name: str, chunks: list[dict]) -> list[str]:
    full_text, page_offsets = build_full_text_with_page_offsets(extract_pages(PDF_DIR / f"{doc_name}.pdf"))
    events = heading_events(full_text, page_offsets)
    offsets = [e[0] for e in events]
    # State after each event: an Item resets the section; a section keeps the
    # Item; a caption ("sub") is appended to the current section until the next
    # section starts, instead of replacing it.
    states, item, section = [], "", ""
    for _, kind, label in events:
        if kind == "item":
            item, section = label, ""
        elif kind == "sub":
            base = section.split(" > ")[0]
            section = f"{base} > {label}" if base else label
        else:
            section = label
        states.append((item, section))

    base = doc_label(doc_name)
    headers = []
    for c in chunks:
        i = bisect.bisect_right(offsets, c["char_start"]) - 1
        item, section = states[i] if i >= 0 else ("", "")
        # Inside a financial statement, note or schedule, that label is what
        # identifies the chunk; the Item can mislead (some filings place the
        # statements after Item 16).
        where = section or item
        headers.append(f"[{base} | page {c['page_start']}" + (f" | {where}]" if where else "]"))
    return headers


def load_chunks_by_doc() -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = {}
    with open(CHUNKS_IN, "rb") as f:
        for line in f:
            c = json.loads(line)
            by_doc.setdefault(c["doc_name"], []).append(c)
    return by_doc


def review_sample(by_doc: dict, headers: dict):
    print("\n=== Spot checks ===")
    for doc, pages in (("VERIZON_2022_10K", (55, 80)), ("AMERICANEXPRESS_2022_10K", (None,)),
                       ("JPMORGAN_2021Q1_10Q", (None,))):
        if doc not in by_doc:
            continue
        for c, h in zip(by_doc[doc], headers[doc]):
            if pages == (None,) and "Parent Company" not in h:
                continue
            if pages != (None,) and c["page_start"] not in pages:
                continue
            print(h, "|", " ".join(c["text"].split())[:70])
    print("\n=== Random sample ===")
    rng = random.Random(3)
    for doc in rng.sample(sorted(headers), 12):
        i = rng.randrange(len(headers[doc]))
        print(headers[doc][i], "|", " ".join(by_doc[doc][i]["text"].split())[:60])
    flat = [h for hs in headers.values() for h in hs]
    with_where = sum(1 for h in flat if h.count("|") >= 4)
    parent = sum(1 for h in flat if "Parent Company" in h)
    print(f"\nchunks: {len(flat)} | with an Item/section label: {with_where} ({with_where / len(flat):.0%}) | "
          f"labelled Parent Company Only: {parent}")


def main():
    sample_only = "--sample" in sys.argv
    by_doc = load_chunks_by_doc()
    docs = sorted(by_doc)
    if sample_only:  # a quick look: the spot-check filings plus a random subset
        keep = {"VERIZON_2022_10K", "AMERICANEXPRESS_2022_10K", "JPMORGAN_2021Q1_10Q"}
        docs = sorted(keep & set(docs)) + random.Random(5).sample([d for d in docs if d not in keep], 20)

    headers = {}
    for n, doc in enumerate(docs, 1):
        headers[doc] = build_headers(doc, by_doc[doc])
        if n % 50 == 0:
            print(f"  {n}/{len(docs)} filings", flush=True)
    review_sample({d: by_doc[d] for d in docs}, headers)
    if sample_only:
        return

    with open(CHUNKS_IN, "rb") as src, open(CHUNKS_OUT, "w", encoding="utf-8") as out:
        cursor = {d: 0 for d in by_doc}
        for line in src:  # original file order == FAISS/BM25 position order
            c = json.loads(line)
            d = c["doc_name"]
            h = headers[d][cursor[d]]
            cursor[d] += 1
            out.write(json.dumps({**c, "header": h, "text": f"{h}\n{c['text']}"}, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(CHUNKS_OUT, "chunks_headed.jsonl")
    print(f"\nWrote {CHUNKS_OUT.name} and {ZIP_OUT.name} ({ZIP_OUT.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
