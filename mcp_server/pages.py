"""
Whole-page reading for `read_page` and `verify_number`.

Project 1 has no "read a page" function: it only stores 2,000-character chunks,
which overlap and cut pages (and tables) in two. So pages are read straight from
the PDFs in data/RAW, with the same PyMuPDF call the chunker used to build those
chunks (page.get_text()) -- the text matches what search returns.

Page numbers are 0-based PDF indexes, the same numbers as in search results,
Project 1's citations and FinanceBench's evidence_page_num. They can differ from
the number printed on the page: 3M_2018_10K page 59 is printed as "60".
"""

import difflib
from pathlib import Path

import pymupdf

MAX_NEIGHBOURS = 2  # 5 pages ~ 6,000 tokens: enough for a table split across pages
PAGE_NOTE = ("Pages are 0-based PDF indexes, as in search results and citations; "
             "the number printed on a page can differ (often by 1).")


class PageError(LookupError):
    """Unknown filing or page out of range. The message says what is valid."""


def clean(text: str) -> str:
    """Drop PyMuPDF's whitespace-only lines and trailing spaces; keeps every word and number."""
    lines = (line.rstrip() for line in text.splitlines())
    return "\n".join(line for line in lines if line.strip())


class PageReader:
    def __init__(self, doc_names: set[str], pdf_dir: Path | None = None):
        if pdf_dir is None:
            # Imported here: the chunker imports fitz, whose deprecation warning goes to
            # STDOUT -- at server start that would corrupt the stdio protocol channel.
            from source.INGESTION.chunker import PDF_DIR as pdf_dir
        self.doc_names = doc_names
        self.pdf_dir = pdf_dir
        self._by_upper = {name.upper(): name for name in doc_names}
        self._page_counts: dict[str, int] = {}

    def resolve_doc(self, doc_name: str) -> str:
        """Exact doc_name for this input (case-insensitive), or PageError with suggestions."""
        name = self._by_upper.get(doc_name.strip().upper())
        if name:
            return name
        close = difflib.get_close_matches(doc_name.strip().upper(), list(self._by_upper), n=3, cutoff=0.8)
        hint = (f" Did you mean: {', '.join(self._by_upper[c] for c in close)}?" if close
                else " Use list_filings(company) to get exact doc_name values.")
        raise PageError(f"Unknown doc_name {doc_name!r}.{hint}")

    def read(self, doc_name: str, page: int, neighbours: int = 0) -> dict:
        name = self.resolve_doc(doc_name)
        if not 0 <= neighbours <= MAX_NEIGHBOURS:
            raise PageError(f"neighbours must be between 0 and {MAX_NEIGHBOURS}.")
        with pymupdf.open(self.pdf_dir / f"{name}.pdf") as pdf:
            count = pdf.page_count
            if not 0 <= page < count:
                raise PageError(f"{name} has pages 0 to {count - 1}; page {page} does not exist.")
            first, last = max(0, page - neighbours), min(count - 1, page + neighbours)
            pages = [{"page": i, "text": clean(pdf[i].get_text())} for i in range(first, last + 1)]
        return {"doc_name": name, "page_count": count, "pages": pages, "note": PAGE_NOTE}

    def page_text(self, doc_name: str, page: int) -> tuple[str, str]:
        """(exact doc_name, text of one page) -- for verify_number."""
        result = self.read(doc_name, page)
        return result["doc_name"], result["pages"][0]["text"]

