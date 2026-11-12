"""search_filings with a fake retriever: no indexes, no model, no Cohere calls."""

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import server
from mcp_server.filings import FilingCatalog
from mcp_server.search import FilingRetriever, NoFilings, search_filings, select_filings
from source.RETRIEVAL.metadata_filter import MetadataFilter

DOC_NAMES = ["3M_2018_10K", "3M_2019_10K", "3M_2023Q2_10Q",
             "PEPSICO_2022_10K", "PEPSICO_2023_8K_dated-2023-05-30"]


class FakeRetriever:
    """Records what it was asked; returns one headed chunk per searched filing."""

    def __init__(self):
        self.metadata_filter = MetadataFilter(DOC_NAMES)
        self.calls = []

    def search_in(self, query, docs, top_k, rerank):
        self.calls.append({"query": query, "docs": docs, "top_k": top_k, "rerank": rerank})
        names = sorted(docs or self.metadata_filter.allowed_docs(query) or DOC_NAMES)[:top_k]
        return [{"rank": i, "doc_name": n, "page_start": 59, "page_end": 60,
                 "text": f"[{n} | page 59]\nPurchases of PP&E (1,577)", "rerank_score": 0.9}
                for i, n in enumerate(names, 1)]


@pytest.fixture
def catalog():
    return FilingCatalog({name: [0] for name in DOC_NAMES})


@pytest.fixture
def fake(catalog):
    retriever = FakeRetriever()
    server.catalog.set(catalog)
    server.retriever.set(retriever)
    yield retriever
    server.catalog.set(None)
    server.retriever.set(None)


@pytest.mark.parametrize("company, year, doc_type, expected", [
    (None, None, None, None),                                    # no filter: Project 1 reads the query
    ("3M", None, None, {"3M_2018_10K", "3M_2019_10K", "3M_2023Q2_10Q"}),
    ("3M", 2018, None, {"3M_2018_10K"}),
    ("3M", None, "10-Q", {"3M_2023Q2_10Q"}),                     # "10-Q" written with a dash
    (None, 2023, "8k", {"PEPSICO_2023_8K_dated-2023-05-30"}),    # no company: across all
])
def test_select_filings(catalog, company, year, doc_type, expected):
    assert select_filings(catalog, company, year, doc_type) == expected


def test_select_filings_explains_what_exists(catalog):
    with pytest.raises(NoFilings, match="Available: 10K 2018, 10K 2019, 10Q 2023"):
        select_filings(catalog, "3M", 2015, None)


def test_explicit_filters_are_passed_to_the_retriever(fake, catalog):
    result = search_filings(fake, catalog, "capex", company="3M", year=2018, top_k=3, rerank=False)
    assert fake.calls == [{"query": "capex", "docs": {"3M_2018_10K"}, "top_k": 3, "rerank": False}]
    assert result["filter"] == "your filters"
    assert result["filings_searched"] == ["3M_2018_10K"]
    assert result["reranked"] is False


def test_without_filters_project1_reads_the_query(fake, catalog):
    result = search_filings(fake, catalog, "What was 3M's FY2018 capex?")
    assert fake.calls[0]["docs"] is None                     # the retriever's own filter decides
    assert result["filter"] == "company/year read from the query"
    assert result["filings_searched"] == ["3M_2018_10K"]


def test_unknown_company_in_query_searches_everything(fake, catalog):
    result = search_filings(fake, catalog, "What is the capital of France?")
    assert result["filter"].startswith("whole corpus")
    assert result["filings_searched"] == "all"


def test_results_are_public_chunks_with_header_split_off(fake, catalog):
    first = search_filings(fake, catalog, "capex", company="3M", year=2018)["results"][0]
    assert first == {"rank": 1, "doc_name": "3M_2018_10K", "page_start": 59, "page_end": 60,
                     "header": "[3M_2018_10K | page 59]", "text": "Purchases of PP&E (1,577)",
                     "rerank_score": 0.9}


@pytest.mark.parametrize("arguments, message", [
    ({"query": "  "}, "empty"),
    ({"query": "capex", "top_k": 0}, "between 1 and 20"),
    ({"query": "capex", "top_k": 21}, "between 1 and 20"),
    ({"query": "capex", "company": "Tesla"}, "No company matches"),
    ({"query": "capex", "company": "3M", "year": 2015}, "No filing matches"),
])
def test_tool_errors_reach_the_model(fake, arguments, message):
    with pytest.raises(ToolError, match=message):
        asyncio.run(server.mcp.call_tool("search_filings", arguments))
    assert fake.calls == []  # rejected before any search (and any Cohere call)


def test_tool_returns_results(fake):
    result = asyncio.run(server.mcp.call_tool("search_filings", {"query": "capex", "company": "pepsi"}))
    data = json.loads(result.content[0].text)
    assert [r["doc_name"] for r in data["results"]] == ["PEPSICO_2022_10K", "PEPSICO_2023_8K_dated-2023-05-30"]


def test_filing_retriever_uses_forced_docs_then_resets():
    """The subclass override, without loading any index."""
    retriever = object.__new__(FilingRetriever)  # skip __init__ (it loads the model)
    retriever._positions_for_docs = lambda docs: sorted(docs)
    seen = []

    def fake_search(query, **kwargs):
        seen.append(retriever._allowed_positions(query))
        raise RuntimeError("Cohere down")

    retriever.search = fake_search
    with pytest.raises(RuntimeError):
        retriever.search_in("q", {"B", "A"}, 5, True)
    assert seen == [["A", "B"]]
    assert retriever._forced_docs is None  # reset even when the search fails
