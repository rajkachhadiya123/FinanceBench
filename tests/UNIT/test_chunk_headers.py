"""Context headers added to every chunk (tools/add_chunk_headers.py)."""
from tools.add_chunk_headers import doc_label, heading_events


def test_doc_label():
    assert doc_label("VERIZON_2022_10K") == "VERIZON | 10-K | FY2022"
    assert doc_label("JOHNSON_JOHNSON_2022Q4_EARNINGS") == "JOHNSON JOHNSON | EARNINGS | FY2022 Q4"


def _events(lines: list[str], pages: int = 1) -> list[tuple[str, str]]:
    text = "\n".join(lines)
    offsets = [0] + [len(text) + i for i in range(1, pages)]
    return [(kind, label) for _off, kind, label in heading_events(text, offsets)]


def test_detects_items_statements_notes_and_parent_company():
    events = _events([
        "Item 8. Financial Statements and Supplementary Data",
        "Consolidated Balance Sheets",
        "Total assets 1,000",
        "Notes to Consolidated Financial Statements",
        "Note 8 Debt",
        "(Parent Company only)",
        "Schedule I - Condensed Financial Information of Registrant",
    ])
    assert events == [
        ("item", "Item 8 Financial Statements and Supplementary Data"),
        ("section", "Consolidated Balance Sheets"),
        ("section", "Notes to Financial Statements"),
        ("section", "Note 8 Debt"),
        ("sub", "Parent Company Only"),
        ("section", "Schedule I - Parent Company Only (Condensed Financial Information of Registrant)"),
    ]


def test_ignores_headings_mentioned_inside_sentences():
    assert _events(["See the Consolidated Balance Sheets included in this report for more detail."]) == []


def test_ignores_table_of_contents_pages():
    toc = ["Item 1. Business", "Item 1A. Risk Factors", "Item 7. Management's Discussion"]
    assert _events(toc) == []
