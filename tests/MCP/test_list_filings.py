"""list_filings tool, on a small fake catalogue (CI has no data/PROCESSED)."""

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import server
from mcp_server.filings import CompanyNotFound, FilingCatalog

# Real FinanceBench name shapes, including the irregular ones.
DOC_NAMES = [
    "3M_2018_10K", "3M_2023Q2_10Q", "3M_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "AMERICANWATERWORKS_2020_10K", "AMERICANWATERWORKS_2021_10K",
    "PEPSICO_2022_10K", "PEPSICO_2023_8K_dated-2023-05-30",
    "JOHNSON_JOHNSON_2022_10K",
    "ACTIVISIONBLIZZARD_2019_10K",
    "MGMRESORTS_2022_10K",
    "BLOCK_2020_10K",
    "MCDONALDS_8K_dated-2023-01-06",
]


@pytest.fixture
def catalog():
    fake = FilingCatalog({name: [0, 1] for name in DOC_NAMES})
    server.catalog.set(fake)
    yield fake
    server.catalog.set(None)


@pytest.mark.parametrize("name, key", [
    ("3M", "3m"),
    ("American Express", "americanexpress"),  # Project 1's filter: full name
    ("AmEx", "americanexpress"),              # Project 1's filter: abbreviation
    ("amex", "americanexpress"),              # new: lowercase letters in order
    ("Pepsi", "pepsico"),                     # new: unique prefix
    ("activision", "activisionblizzard"),     # new: unique prefix
    ("mgm", "mgmresorts"),
    ("J&J", "johnsonjohnson"),
    ("Square", "block"),                      # Project 1's rename table
    ("McDonald's", "mcdonalds"),
])
def test_resolve_company(catalog, name, key):
    assert catalog.resolve_company(name) == key


def test_ambiguous_name_lists_candidates_instead_of_guessing(catalog):
    with pytest.raises(CompanyNotFound) as exc:
        catalog.resolve_company("American")
    assert exc.value.candidates == ["AMERICANEXPRESS", "AMERICANWATERWORKS"]


def test_unknown_name_suggests_close_spellings(catalog):
    with pytest.raises(CompanyNotFound) as exc:
        catalog.resolve_company("Pepsicp")  # typo, not a superstring
    assert exc.value.candidates == ["PEPSICO"]
    with pytest.raises(CompanyNotFound, match="No company matches") as exc:
        catalog.resolve_company("Tesla")  # not in FinanceBench: no far-fetched guess
    assert exc.value.candidates == []


def test_filings_are_sorted_and_typed(catalog):
    result = catalog.filings_of("3M")
    assert [f["doc_name"] for f in result["filings"]] == ["3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q"]
    assert result["filings"][2] == {"doc_name": "3M_2023Q2_10Q", "type": "10Q",
                                    "fiscal_year": 2023, "quarter": "Q2", "filed": None}
    eight_k = catalog.filings_of("McDonalds")["filings"][0]
    assert (eight_k["type"], eight_k["fiscal_year"], eight_k["filed"]) == ("8K", 2023, "2023-01-06")


def test_summary_has_one_row_per_company(catalog):
    rows = {r["company"]: r for r in catalog.summary()}
    assert len(rows) == 9
    assert rows["3M"] == {"company": "3M", "filings": 3, "types": ["10K", "10Q"],
                          "fiscal_years": "2018-2023"}


def call(arguments):
    return asyncio.run(server.mcp.call_tool("list_filings", arguments))


def test_tool_without_company_returns_summary(catalog):
    data = json.loads(call({}).content[0].text)
    assert len(data["companies"]) == 9


def test_tool_with_company_returns_doc_names(catalog):
    data = json.loads(call({"company": "pepsi"}).content[0].text)
    assert data["company"] == "PEPSICO"
    assert [f["doc_name"] for f in data["filings"]] == ["PEPSICO_2022_10K", "PEPSICO_2023_8K_dated-2023-05-30"]


def test_tool_error_names_the_candidates(catalog):
    with pytest.raises(ToolError, match="AMERICANEXPRESS, AMERICANWATERWORKS"):
        call({"company": "American"})
