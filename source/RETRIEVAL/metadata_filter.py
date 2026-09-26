"""
Extracts the company and fiscal year a question is asking about, and maps them
onto the filings that could answer it.

Why this exists: the corpus holds ~9 near-identical filings per company (3M's
2018 and 2021 10-Ks share most of their wording), so semantic similarity
cannot tell them apart. Measured on the baseline, 61.3% of questions retrieved
from the wrong document entirely. The distinguishing information is the
document's identity, which lives in the question text, not in the prose.

Safety principle: a WRONG filter guarantees failure, while NO filter merely
falls back to baseline behaviour. So anything uncertain returns None and the
caller searches the whole corpus.
"""
import re

YEAR_IN_NAME_RE = re.compile(r"^(20\d\d)(Q\d)?$")
ANY_YEAR_RE = re.compile(r"20[0-2]\d")
# "FY2018", "FY 2018", "fiscal 2018", "Q2 of FY2023", or a bare "2018"
QUESTION_YEAR_RE = re.compile(r"(?:FY\s*|fiscal\s+)?(20[0-2]\d)", re.IGNORECASE)

# Short names are matched on word boundaries instead of as substrings, because
# a 2-3 character substring hits far too much unrelated text.
SHORT_NAME_MAX_LEN = 4

# Question wording that doesn't match any filename token.
ALIASES = {
    "amex": "americanexpress",
    "jnj": "johnsonjohnson",
    "jj": "johnsonjohnson",
    "jandj": "johnsonjohnson",
    "activision": "activisionblizzard",
    "cvs": "cvshealth",
    "coke": "cocacola",
    "pgande": "pge",
    "square": "block",  # Block was renamed from Square in 2021
}

# Filename typos in the source dataset that refer to a company we already know.
FILENAME_TYPOS = {
    "activsionblizzard": "activisionblizzard",
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_doc_name(doc_name: str) -> tuple[str, int | None, str | None, str]:
    """
    Split a filing name into (company_key, year, quarter, doc_type).

    Handles the regular shape (3M_2018_10K, ADOBE_2022Q2_10Q) and the handful
    of irregular ones (MCDONALDS_8K_dated-2023-01-06) where the year is only
    present inside a date string.
    """
    parts = doc_name.split("_")
    for i, part in enumerate(parts):
        match = YEAR_IN_NAME_RE.match(part)
        if match:
            company = normalize("_".join(parts[:i]))
            return company, int(match.group(1)), match.group(2), "_".join(parts[i + 1:])

    # Irregular name: fall back to the first 4-digit year found anywhere.
    fallback = ANY_YEAR_RE.search(doc_name)
    year = int(fallback.group()) if fallback else None
    company = normalize(parts[0])
    return company, year, None, "_".join(parts[1:])


class MetadataFilter:
    def __init__(self, doc_names: list[str]):
        self.docs_by_company: dict[str, list[str]] = {}
        self.year_by_doc: dict[str, int | None] = {}

        for doc_name in doc_names:
            company, year, _quarter, _doc_type = parse_doc_name(doc_name)
            company = FILENAME_TYPOS.get(company, company)
            self.docs_by_company.setdefault(company, []).append(doc_name)
            self.year_by_doc[doc_name] = year

        # Longest first, so "americanexpress" wins over a shorter overlapping key.
        self.company_keys = sorted(self.docs_by_company, key=len, reverse=True)

    def identify_company(self, question: str) -> str | None:
        normalized_q = normalize(question)

        for key in self.company_keys:
            if len(key) <= SHORT_NAME_MAX_LEN:
                # e.g. 3M / AMD / AES -- require a standalone word in the raw text
                if re.search(rf"\b{re.escape(key)}\b", question, re.IGNORECASE):
                    return key
            elif key in normalized_q:
                return key

        for alias, company in ALIASES.items():
            if company not in self.docs_by_company:
                continue
            if len(alias) <= SHORT_NAME_MAX_LEN:
                if re.search(rf"\b{re.escape(alias)}\b", question, re.IGNORECASE):
                    return company
            elif alias in normalized_q:
                return company

        return None

    @staticmethod
    def extract_years(question: str) -> set[int]:
        return {int(y) for y in QUESTION_YEAR_RE.findall(question)}

    def allowed_docs(self, question: str, use_year: bool = True) -> set[str] | None:
        """
        Filings that could answer this question, or None to search everything.

        Every mentioned year is kept, not just the latest: a question about the
        FY2015-to-FY2016 change is answered by the FY2016 filing (which reports
        both years), so narrowing to one year would risk excluding the answer.
        """
        company = self.identify_company(question)
        if company is None:
            return None  # can't identify the company -- don't guess

        candidates = self.docs_by_company[company]
        if not use_year:
            return set(candidates)

        years = self.extract_years(question)
        if not years:
            return set(candidates)

        year_matched = {d for d in candidates if self.year_by_doc[d] in years}
        # An empty year match means our year reading was wrong, not that the
        # company has no relevant filings -- fall back rather than return nothing.
        return year_matched or set(candidates)
