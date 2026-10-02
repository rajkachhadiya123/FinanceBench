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
# Two-digit fiscal shorthand: "FY20", "FY21". The (?!\d) guard stops this from
# matching the "FY20" inside "FY2018". Missing this caused a Pfizer question
# asking about "FY20 and FY21" to extract no year at all, leaving all nine
# Pfizer filings in scope.
SHORT_YEAR_RE = re.compile(r"FY\s?'?(\d{2})(?!\d)", re.IGNORECASE)

# Short names are matched on word boundaries instead of as substrings, because
# a 2-3 character substring hits far too much unrelated text.
SHORT_NAME_MAX_LEN = 4

# Abbreviations are resolved against the company names already present in the
# corpus filenames, NOT from a hand-maintained dictionary. A dictionary grows
# one entry per failed question, which fits the benchmark rather than the task:
# drop a new company's filings into data/RAW/ and a dictionary would not know
# it, while corpus-derived matching works with no code change.
#
# The only hardcoded list is finance vocabulary that must never be mistaken for
# a company name. It is domain vocabulary, not per-question patching -- "EPS"
# would otherwise match PEPSICO as an in-order subsequence (p-E-P-S-ico).
FINANCE_JARGON = {
    "eps", "ebit", "ebitda", "roa", "roe", "roi", "roic", "cagr", "yoy", "qoq",
    "capex", "opex", "ppne", "ppe", "gaap", "sec", "usd", "fy", "ttm", "wacc",
    "fcf", "dcf", "cogs", "sga", "da", "nci", "eoy", "ytd", "mda", "npv",
    "q1", "q2", "q3", "q4", "10k", "10q", "8k", "jv", "ma", "ceo", "cfo",
}
# Candidate abbreviations: short word-like tokens carrying two or more capitals.
# Requiring 2+ capitals (rather than all-caps) catches mixed forms like "JnJ"
# while excluding ordinary capitalised words such as "What" or "Does".
ABBREVIATION_RE = re.compile(r"\b([A-Za-z&.]{2,8})\b")
MIN_CAPITALS = 2

# Companies renamed since filing, which no string match can bridge.
RENAMES = {"square": "block"}

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
        """
        Resolve the company by full name first, then by abbreviation matched
        against corpus company names. Returns None when uncertain -- a wrong
        filter guarantees failure, while no filter only forgoes the speedup.
        """
        normalized_q = normalize(question)

        # 1. Full name appears in the question (handles "American Express",
        #    "Johnson & Johnson", "Coca Cola" via normalization).
        for key in self.company_keys:
            if len(key) <= SHORT_NAME_MAX_LEN:
                # 3M / AMD / AES -- need a standalone word, not a substring
                if re.search(rf"\b{re.escape(key)}\b", question, re.IGNORECASE):
                    return key
            elif key in normalized_q:
                return key

        # 2. Former name used instead of the current one.
        for old, new in RENAMES.items():
            if old in normalized_q and new in self.docs_by_company:
                return new

        # 3. Abbreviation resolved against corpus names, accepted only when it
        #    matches exactly one company.
        return self._resolve_abbreviation(question)

    def _resolve_abbreviation(self, question: str) -> str | None:
        for raw in ABBREVIATION_RE.findall(question):
            if sum(1 for c in raw if c.isupper()) < MIN_CAPITALS:
                continue  # an ordinary word, not an abbreviation
            token = normalize(raw)
            if len(token) < 2 or token in FINANCE_JARGON:
                continue

            # Prefix beats subsequence: "CVS" prefixes CVSHEALTH, but is also a
            # loose subsequence of aCtiVisionblizzard. Without this ordering the
            # spurious second match made the pair ambiguous and both were lost.
            prefix = {k for k in self.company_keys if k.startswith(token)}
            candidates = prefix or {
                k for k in self.company_keys if self._is_subsequence(token, k)
            }
            # Ambiguity means we cannot be confident, so we decline rather than
            # guess -- the caller then searches the whole corpus.
            if len(candidates) == 1:
                return candidates.pop()
        return None

    @staticmethod
    def _is_subsequence(short: str, full: str) -> bool:
        """Do short's letters appear in order within full? ("amex" in "americanexpress")"""
        it = iter(full)
        return all(char in it for char in short)

    @staticmethod
    def extract_years(question: str) -> set[int]:
        years = {int(y) for y in QUESTION_YEAR_RE.findall(question)}
        years |= {2000 + int(y) for y in SHORT_YEAR_RE.findall(question)}
        return years

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
