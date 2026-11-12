"""read_page and verify_number, on tiny PDFs made here (CI has no data/RAW)."""

import asyncio
import json

import pymupdf
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import server
from mcp_server.pages import PageError, PageReader, clean
from mcp_server.verify import find_value

# Page 1 mimics a PyMuPDF cash-flow table: label line, then one line per year.
PAGES = [
    "Cover page\n \n3M Company",
    "Cash Flows from Investing Activities\n \nPurchases of property, plant and equipment (PP&E)\n"
    "(1,577)\n(1,373)\n(1,420)\nDividends paid to shareholders\n(3,193)\n \n60",
    "Total assets\n36,500\n37,987",
]


@pytest.fixture
def reader(tmp_path):
    with pymupdf.open() as pdf:
        for text in PAGES:
            pdf.new_page().insert_text((72, 72), text)
        pdf.save(tmp_path / "3M_2018_10K.pdf")
    fake = PageReader({"3M_2018_10K"}, pdf_dir=tmp_path)
    server.pages.set(fake)
    yield fake
    server.pages.set(None)


def test_clean_drops_blank_lines_but_keeps_every_number():
    assert clean("A \n \n(1,577) \n\n  \n60") == "A\n(1,577)\n60"


def test_read_one_page(reader):
    result = reader.read("3M_2018_10K", 1)
    assert result["page_count"] == 3
    assert [p["page"] for p in result["pages"]] == [1]
    assert "(1,577)" in result["pages"][0]["text"]
    assert "0-based" in result["note"]


def test_neighbours_are_clipped_at_the_ends(reader):
    assert [p["page"] for p in reader.read("3M_2018_10K", 0, neighbours=1)["pages"]] == [0, 1]
    assert [p["page"] for p in reader.read("3M_2018_10K", 1, neighbours=2)["pages"]] == [0, 1, 2]


def test_doc_name_is_case_insensitive(reader):
    assert reader.read("3m_2018_10k", 0)["doc_name"] == "3M_2018_10K"


@pytest.mark.parametrize("doc, page, neighbours, message", [
    ("3M_2018_10K", 3, 0, "pages 0 to 2"),
    ("3M_2018_10K", -1, 0, "pages 0 to 2"),
    ("3M_2018_10K", 0, 3, "between 0 and 2"),
    ("3M_2018_10Q", 0, 0, "Did you mean: 3M_2018_10K"),
    ("Apple_2018_10K", 0, 0, "list_filings"),
])
def test_bad_requests_say_what_is_valid(reader, doc, page, neighbours, message):
    with pytest.raises(PageError, match=message):
        reader.read(doc, page, neighbours)


TABLE = PAGES[1]


@pytest.mark.parametrize("value", [1577, -1577, 1.577, 1577000, 1.58])
def test_value_found_across_sign_units_and_rounding(value):
    result = find_value(value, TABLE)
    assert result["found"] and result["matches"][0]["figure"] == "1,577"


def test_match_context_shows_the_row_label():
    match = find_value(1577, TABLE)["matches"][0]
    assert "Purchases of property, plant and equipment" in match["context"]
    assert match["same_units"] is True


def test_value_not_on_page():
    result = find_value(1500, TABLE)
    assert result == {"found": False, "matches": [], "total_matches": 0, "rule": result["rule"]}


def test_small_numbers_are_checked_too():
    # Project 1's extract_figures skips integers <= 12 and years; here they count.
    assert find_value(60, TABLE)["found"] is True


def call(name, arguments):
    return json.loads(asyncio.run(server.mcp.call_tool(name, arguments)).content[0].text)


def test_read_page_tool(reader):
    result = call("read_page", {"doc_name": "3M_2018_10K", "page": 2})
    assert result["pages"][0]["text"] == "Total assets\n36,500\n37,987"


def test_verify_number_tool(reader):
    result = call("verify_number", {"value": 3193, "doc_name": "3m_2018_10k", "page": 1})
    assert result["found"] is True and result["doc_name"] == "3M_2018_10K"
    assert "Dividends paid" in result["matches"][0]["context"]


def test_tools_report_bad_pages_to_the_model(reader):
    for name, args in (("read_page", {"doc_name": "3M_2018_10K", "page": 9}),
                       ("verify_number", {"value": 1, "doc_name": "3M_2018_10K", "page": 9})):
        with pytest.raises(ToolError, match="pages 0 to 2"):
            asyncio.run(server.mcp.call_tool(name, args))
