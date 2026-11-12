"""
The filing catalogue behind `list_filings` (and, later, name checks in the other tools).

Built from Project 1, unchanged:
  - the filing names come from the chunk metadata the retriever searches
    (chunk_meta_headed.jsonl), so a filing listed here is always searchable;
  - catalogue() (source/API/pipeline.py) groups them by company, the same
    grouping the API's GET /documents returns;
  - MetadataFilter.identify_company() resolves names like "AmEx" or "J&J".

What is new here: a gentler name lookup. The metadata filter is deliberately
strict, because it reads whole questions and a wrong guess there guarantees a
wrong answer. A tool receives just a company name, so after the filter we also
try a unique prefix ("pepsi" -> PepsiCo) and a unique in-order letter match
("amex" -> American Express). Anything still ambiguous is reported, not guessed.
"""

import difflib
import json
from pathlib import Path

from source.INGESTION.embedder3 import HEADED_META_PATH
from source.RETRIEVAL.metadata_filter import MetadataFilter, normalize

MIN_FUZZY_LENGTH = 3  # "am" would match half the corpus
# Spelling-suggestion threshold (difflib ratio). At 0.6 "Tesla" suggested INTEL;
# 0.75 still catches one-letter typos such as "Pepsicp" -> PEPSICO (0.86).
SUGGEST_CUTOFF = 0.75


class CompanyNotFound(LookupError):
    """No single company matches. `candidates` holds the company names to suggest."""

    def __init__(self, name: str, candidates: list[str]):
        self.candidates = candidates
        if candidates:
            message = f"No single company matches {name!r}. Did you mean: {', '.join(candidates)}?"
        else:
            message = f"No company matches {name!r}. Call list_filings() without a company to see all."
        super().__init__(message)


class FilingCatalog:
    def __init__(self, positions_by_doc: dict[str, list[int]]):
        # Imported here, not at the top: source.API.pipeline loads torch via the
        # retriever (~10 s), and the MCP server should start instantly.
        from source.API.pipeline import catalogue

        self.companies = {entry["key"]: entry for entry in catalogue(positions_by_doc)}
        self.doc_names = set(positions_by_doc)
        self._filter = MetadataFilter(list(positions_by_doc))

    @classmethod
    def from_meta_file(cls, path: Path = HEADED_META_PATH) -> "FilingCatalog":
        positions_by_doc: dict[str, list[int]] = {}
        with open(path, "rb") as f:
            for position, line in enumerate(f):
                positions_by_doc.setdefault(json.loads(line)["doc_name"], []).append(position)
        return cls(positions_by_doc)

    def resolve_company(self, name: str) -> str:
        """Company key for a user-supplied name, or CompanyNotFound."""
        key = self._filter.identify_company(name)
        if key in self.companies:
            return key

        wanted = normalize(name)
        if wanted in self.companies:
            return wanted
        if len(wanted) >= MIN_FUZZY_LENGTH:
            for matches in (
                [k for k in self.companies if k.startswith(wanted)],
                [k for k in self.companies if _letters_in_order(wanted, k)],
            ):
                if len(matches) == 1:
                    return matches[0]
                if matches:
                    raise CompanyNotFound(name, self._names(matches))

        close = difflib.get_close_matches(wanted, list(self.companies), n=3, cutoff=SUGGEST_CUTOFF)
        raise CompanyNotFound(name, self._names(close))

    def summary(self) -> list[dict]:
        """One row per company: what exists, without listing every filing."""
        rows = []
        for entry in self.companies.values():
            filings = entry["filings"]
            years = [f["fiscal_year"] for f in filings if f["fiscal_year"]]
            rows.append({
                "company": entry["company"],
                "filings": len(filings),
                "types": sorted({f["type"] for f in filings}),
                "fiscal_years": f"{min(years)}-{max(years)}" if years else None,
            })
        return rows

    def filings_of(self, name: str) -> dict:
        entry = self.companies[self.resolve_company(name)]
        return {
            "company": entry["company"],
            "filings": [
                {k: f[k] for k in ("doc_name", "type", "fiscal_year", "quarter", "filed")}
                for f in entry["filings"]
            ],
        }

    def _names(self, keys: list[str]) -> list[str]:
        return sorted(self.companies[k]["company"] for k in keys)


def _letters_in_order(short: str, full: str) -> bool:
    """"amex" -> a..m..e..x appear in order in "americanexpress"."""
    it = iter(full)
    return all(char in it for char in short)
