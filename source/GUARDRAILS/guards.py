"""
Rule-based guardrails for FilingLens (no LLM calls).

Two principles, because a guard that blocks a real finance question is worse than
no guard:
  * Input guards BLOCK only clear abuse: hidden characters, unreadable text, and
    narrow prompt-injection patterns. Everything else is at most a warning.
  * Output guards never change the answer. They only add signals next to it:
    how many of the answer's figures appear in the retrieved text, how relevant
    the best retrieved chunk was, and a disclaimer for investment-advice requests.

Every pattern was checked against all 150 FinanceBench questions (none blocked) and
the output signals against the 150 graded answers -- tools/evaluate_guardrails.py.
"""
import re
import unicodedata

GUARDRAILS_VERSION = "g1"  # part of the API's cache key: changing guards re-computes answers

# --- input ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    # "ignore / disregard ... previous instructions" -- needs both the verb and an
    # instruction noun, so "ignoring one-time items" or "excluding the impact" pass.
    re.compile(r"\b(ignore|disregard|forget|override|bypass)\b[^.?!\n]{0,40}"
               r"\b(previous|prior|above|earlier|all|any|your|the|these|system)\b[^.?!\n]{0,20}"
               r"\b(instructions?|prompts?|rules|directions|guidelines)\b", re.I),
    # attempts to extract the prompt
    re.compile(r"\b(reveal|show|print|repeat|output|display|leak|tell me)\b[^.?!\n]{0,30}"
               r"\b(system prompt|your (prompt|instructions|rules)|hidden (prompt|instructions)"
               r"|initial prompt)\b", re.I),
    # role-play takeovers
    re.compile(r"\b(you are now|from now on,? you|pretend (to be|you are)"
               r"|act as an? (unrestricted|uncensored|unfiltered|different|new)\b)", re.I),
    re.compile(r"\b(jailbreak|developer mode|do anything now)\b", re.I),
    re.compile(r"\bDAN\b"),  # case-sensitive: the "Do Anything Now" persona, not a name
    # chat-template / role tags smuggled into the question
    re.compile(r"<\s*/?\s*(system|assistant|user|im_start|im_end)\s*>|\[/?INST\]"
               r"|###\s*(system|instruction)|<\|[a-z_]+\|>", re.I),
]
_HIDDEN_CHARS = re.compile("[​-\u200F\u202A-\u202E⁠-⁤\u2066-\u2069﻿]")
MIN_LETTER_SHARE = 0.4     # share of non-space characters that are letters
MAX_CHAR_RUN = 15          # e.g. "!!!!!!!!!!!!!!!!" or "aaaaaaaaaaaaaaaa"
MAX_TOKEN_SHARE = 0.5      # one word making up over half of a 10+ word question

_ADVICE_PATTERNS = re.compile(
    r"\bshould (i|we|you|one) (buy|sell|invest|short|hold|purchase)\b"
    r"|\b(is|are|would)\b[^.?!\n]{0,40}\bgood (investment|buy|stock|time to (buy|sell|invest))\b"
    r"|\bworth (buying|investing)\b|\b(buy|sell|hold) (recommendation|rating)\b"
    r"|\bprice target\b|\binvest (my|our) money\b", re.I)
DISCLAIMER = ("FilingLens reports figures from SEC filings; this is not investment advice "
              "or a recommendation to buy or sell any security.")


def check_input(question: str) -> dict:
    """{"blocked": bool, "reason": str | None, "warnings": [...]} -- never calls an API."""
    def blocked(reason):
        return {"blocked": True, "reason": reason, "warnings": []}

    if _HIDDEN_CHARS.search(question) or any(
            unicodedata.category(ch) == "Cc" and ch not in "\n\t\r" for ch in question):
        return blocked("The question contains hidden or control characters.")
    visible = [ch for ch in question if not ch.isspace()]
    if visible and sum(ch.isalpha() for ch in visible) / len(visible) < MIN_LETTER_SHARE:
        return blocked("The question is not readable text.")
    if re.search(r"(.)\1{%d,}" % (MAX_CHAR_RUN - 1), question):
        return blocked("The question contains a long run of repeated characters.")
    words = re.findall(r"\w+", question.lower())
    if len(words) >= 10 and max(words.count(w) for w in set(words)) / len(words) > MAX_TOKEN_SHARE:
        return blocked("The question repeats the same word too often.")
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(question):
            return blocked("The question looks like an attempt to change the assistant's instructions.")
    return {"blocked": False, "reason": None, "warnings": []}


def needs_disclaimer(question: str) -> bool:
    return bool(_ADVICE_PATTERNS.search(question))


# --- output ----------------------------------------------------------------------------
# Numbers, excluding ones glued to letters (FY2022, Q4, v2, 10-K's "10" is caught below).
_NUMBER = re.compile(r"(?<![A-Za-z\d.,])\d[\d,]*(?:\.\d+)?(?![\d]|-[KQ]\b)")
_SCALES = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)  # "$1.58 billion" vs "1,577" (millions)
SMALL_INT = 12  # list markers, quarters, months: too common to be evidence
# Page references ("page 55-56", "Page: 45", "pp. 12") and the "x 100" of a percentage
# formula are not financial figures; left in, they were most of the false alarms.
_NOT_FIGURES = re.compile(r"\b(pages?|pp?\.)\s*:?\s*\d+(\s*[-‐-―]\s*\d+)?|[×x*]\s*100\b", re.I)
# Best Cohere relevance below this -> low_confidence. Measured: every answered
# benchmark question scored >= 0.816; out-of-scope questions (a company not in the
# data, a recipe, sport) scored 0.28-0.39.
LOW_CONFIDENCE = 0.50
# Share of figures found in the sources below this -> grounded False. Calculated
# results (growth rates, ratios) never appear in a filing, so correct calculation
# answers measured 0.40-0.47 at worst; 0.35 flags only answers whose figures are
# mostly untraceable (0 of 111 correct answers on FinanceBench).
MIN_GROUNDING = 0.35


def extract_figures(text: str) -> list[float]:
    figures = []
    for match in _NUMBER.finditer(text):
        raw = match.group().rstrip(",")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if value.is_integer() and (1990 <= value <= 2035 or value <= SMALL_INT):
            continue  # years and small counters carry no evidence
        figures.append(value)
    return figures


def _decimals(value: float) -> int:
    text = repr(value)
    return len(text.split(".")[1]) if "." in text and not text.endswith(".0") else 0


def _found(value: float, sources: set[float]) -> bool:
    places = _decimals(value)
    for source in sources:
        for scale in _SCALES:
            scaled = source * scale
            if abs(value - scaled) <= 0.005 * max(value, scaled):
                return True
            if places and round(scaled, places) == value:  # source rounded the way the answer is
                return True
    return False


def check_output(answer: str, reasoning: str, chunks: list[dict], not_found: bool) -> dict:
    """
    Signals about an answer; the answer itself is never modified.

    grounding_ratio -- share of the figures in the answer and its reasoning that also
                       appear in the retrieved chunks (sign ignored; unit scaling and
                       rounding allowed). Calculated results (ratios, growth rates)
                       legitimately do not appear, so this is a signal, not proof.
    retrieval_confidence -- best Cohere relevance score among the chunks used.
    """
    scores = [c["rerank_score"] for c in chunks if c.get("rerank_score") is not None]
    confidence = round(max(scores), 4) if scores else None
    result = {
        "grounded": None, "grounding_ratio": None, "figures_checked": 0, "ungrounded_figures": [],
        "retrieval_confidence": confidence,
        "low_confidence": confidence is not None and confidence < LOW_CONFIDENCE,
    }
    if not_found:
        return result

    sources = {abs(v) for c in chunks for v in extract_figures(c["text"])}
    figures = [abs(v) for v in extract_figures(_NOT_FIGURES.sub(" ", f"{answer}\n{reasoning}"))]
    if not figures:
        return result
    missing = [v for v in dict.fromkeys(figures) if not _found(v, sources)]
    unique = len(dict.fromkeys(figures))
    ratio = round(1 - len(missing) / unique, 3)
    result.update({
        "grounded": ratio >= MIN_GROUNDING, "grounding_ratio": ratio, "figures_checked": unique,
        "ungrounded_figures": [int(v) if v.is_integer() else v for v in missing[:10]],
    })
    return result
