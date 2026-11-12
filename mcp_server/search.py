"""
The search behind `search_filings`: Project 1's final retrieval, with explicit filters.

Project 1's Retriever.search() only filters by what it can read out of the query
text (company + fiscal year). A tool should also accept them as arguments, so
FilingRetriever (a subclass; Project 1 is not modified) lets the caller name the
filings to search. Everything else is Project 1's pipeline, step for step:

    filter -> Qwen3 (headed chunks) top 20 + BM25 top 20 -> RRF -> Cohere rerank -> top k

  company / year / doc_type given   -> search exactly those filings
  none given                        -> Project 1's metadata filter reads the query
  query names no known company      -> the whole corpus
"""

from source.API.pipeline import public_chunk
from source.RETRIEVAL.retriever import Retriever

from mcp_server.filings import FilingCatalog

MAX_TOP_K = 20
SHOW_FILINGS = 12  # list the searched filings by name up to this many


class NoFilings(LookupError):
    """The filters exclude every filing."""


class FilingRetriever(Retriever):
    """Retriever that can be told which filings to search, instead of guessing from the query."""

    _forced_docs: set[str] | None = None

    def _allowed_positions(self, query):
        if self._forced_docs is not None:
            return self._positions_for_docs(self._forced_docs)
        return super()._allowed_positions(query)

    def search_in(self, query: str, docs: set[str] | None, top_k: int, rerank: bool) -> list[dict]:
        self._forced_docs = docs
        try:
            return self.search(query, top_k=top_k, metadata_filter=True, rerank=rerank)
        finally:
            self._forced_docs = None


def load_retriever() -> FilingRetriever:
    # Project 1 prints progress ("Loading indexes ..."). That is safe only after the
    # MCP SDK has taken over stdout (it then sends stray prints to stderr), which is
    # why the server loads this lazily / from its lifespan, never at import.
    return FilingRetriever(use_reranker="cohere", embedding="qwen_headed")


def select_filings(catalog: FilingCatalog, company: str | None, year: int | None,
                   doc_type: str | None) -> set[str] | None:
    """Filings matching the explicit filters, or None when no filter was given."""
    if not company and year is None and not doc_type:
        return None
    if company:
        entries = [catalog.companies[catalog.resolve_company(company)]]
    else:
        entries = list(catalog.companies.values())
    filings = [f for e in entries for f in e["filings"]]
    wanted_type = doc_type.strip().upper().replace("-", "") if doc_type else None
    chosen = {f["doc_name"] for f in filings
              if (year is None or f["fiscal_year"] == year)
              and (wanted_type is None or f["type"] == wanted_type)}
    if not chosen:
        available = sorted({(f["type"], f["fiscal_year"]) for f in filings}, key=lambda t: (t[1] or 0, t[0]))
        shown = ", ".join(f"{t} {y}" for t, y in available[:20])
        raise NoFilings(f"No filing matches company={company!r}, year={year}, doc_type={doc_type!r}. "
                        f"Available: {shown}.")
    return chosen


def search_filings(retriever, catalog: FilingCatalog, query: str, company: str | None = None,
                   year: int | None = None, doc_type: str | None = None, top_k: int = 5,
                   rerank: bool = True) -> dict:
    if not query.strip():
        raise ValueError("query is empty.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    docs = select_filings(catalog, company, year, doc_type)
    if docs is not None:
        how, searched = "your filters", docs
    else:
        searched = retriever.metadata_filter.allowed_docs(query)  # what Project 1 will pick
        how = "company/year read from the query" if searched else "whole corpus (no company recognised)"

    chunks = retriever.search_in(query, docs, top_k, rerank)
    return {
        "filter": how,
        "filings_searched": (sorted(searched) if searched and len(searched) <= SHOW_FILINGS
                             else len(searched) if searched else "all"),
        "reranked": rerank,
        "results": [public_chunk(c) for c in chunks],
    }
