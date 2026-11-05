"""Parsing of the LLM's reply and the v4 formula check (no API calls)."""
from source.GENERATION.answer_generator import (
    CITATIONS_RE,
    FINAL_ANSWER_RE,
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    AnswerGenerator,
    check_formula,
)

REPLY = """REASONING: Revenues 51,217 (10-K p54).
FINAL ANSWER: Revenue was $51,217 million.
- product A
- product B
CITATIONS: NIKE_2023_10K | Page 54
"""


def test_active_prompt_is_v2():
    assert PROMPT_VERSION == "v2"
    assert "{context}" in PROMPT_TEMPLATE and "{question}" in PROMPT_TEMPLATE


def test_final_answer_keeps_multi_line_lists():
    answer = AnswerGenerator._extract(FINAL_ANSWER_RE, REPLY)
    assert answer.startswith("Revenue was $51,217 million.") and "- product B" in answer
    assert AnswerGenerator._extract(CITATIONS_RE, REPLY) == "NIKE_2023_10K | Page 54"
    assert AnswerGenerator._extract_reasoning(REPLY) == "Revenues 51,217 (10-K p54)."


def test_build_context_labels_each_chunk():
    context = AnswerGenerator.build_context([
        {"doc_name": "NIKE_2023_10K", "page_start": 54, "page_end": 54, "text": "a"},
        {"doc_name": "NIKE_2023_10K", "page_start": 55, "page_end": 56, "text": "b"},
    ])
    assert "[Document: NIKE_2023_10K | Page: 54]" in context
    assert "[Document: NIKE_2023_10K | Page: 55-56]" in context


def test_check_formula():
    assert check_formula("(11,512 + 2,763) / 86,392 * 100", "16.5%") == "match"
    assert check_formula("(31) / (5,022) * 100", "0.6%") == "match"      # rounded answer
    assert check_formula("(1,250 + 300) / 9,800", "51.1%") == "mismatch"
    assert check_formula("none", "Yes") == "n/a"
    assert check_formula("Revenue / Assets", "2.1") == "n/a"             # not plain arithmetic
