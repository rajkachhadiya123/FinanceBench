"""
The check behind `verify_number`: does this value appear on that page?

Reuses Project 1's grounding check (source/GUARDRAILS/guards.py), so a figure
counts as present exactly when the API's output check would count it:
  - _NUMBER finds the numbers in the text ("1,577", "(1,577)", "8.7");
  - _found() compares with 0.5% tolerance, across units and rounding:
        1577 on the page  <->  1577, 1.577 (billions), 1,577,000 (thousands), 1.58
The sign is ignored, because filings print outflows as "(1,577)".

One difference: Project 1's extract_figures() skips years and integers <= 12
(noise when scanning a whole answer). Here the caller asks about one specific
value, so every number on the page is a candidate.
"""

import re

from source.GUARDRAILS.guards import _NUMBER, _found

MAX_MATCHES = 5
CONTEXT_BEFORE, CONTEXT_AFTER = 90, 25  # table rows are linear text: the label comes first
RULE = ("Sign ignored (filings print outflows as (1,577)); 0.5% tolerance; units may differ "
        "(1.577 billion matches 1,577 million); a rounded value matches its source.")


def find_value(value: float, text: str) -> dict:
    """Matches of value on this page, each with the text just before it (its row label)."""
    target = abs(value)
    matches = []
    for match in _NUMBER.finditer(text):
        raw = match.group().rstrip(",")
        try:
            figure = float(raw.replace(",", ""))
        except ValueError:
            continue
        if not _found(target, {figure}):
            continue
        start = max(0, match.start() - CONTEXT_BEFORE)
        context = re.sub(r"\s+", " ", text[start:match.end() + CONTEXT_AFTER]).strip()
        matches.append({"figure": raw, "same_units": figure == target,
                        "context": context})

    # Same-units matches first: 5 vs "5,000" (thousands) is a weaker match than 5 vs "5".
    matches.sort(key=lambda m: not m["same_units"])
    return {"found": bool(matches), "matches": matches[:MAX_MATCHES],
            "total_matches": len(matches), "rule": RULE}
