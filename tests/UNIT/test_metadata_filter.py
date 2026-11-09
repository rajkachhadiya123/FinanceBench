"""Company + fiscal-year extraction that decides which filings are searched."""
import pytest

from source.RETRIEVAL.metadata_filter import MetadataFilter, parse_doc_name


def test_parse_doc_name_regular_and_irregular():
    assert parse_doc_name("3M_2018_10K") == ("3m", 2018, None, "10K")
    assert parse_doc_name("JOHNSON_JOHNSON_2022Q4_EARNINGS") == ("johnsonjohnson", 2022, "Q4", "EARNINGS")
    company, year, _, _ = parse_doc_name("MCDONALDS_8K_dated-2023-01-06")
    assert (company, year) == ("mcdonalds", 2023)


def test_full_name_and_year_select_one_filing(doc_names):
    f = MetadataFilter(doc_names)
    assert f.allowed_docs("What was Verizon's capex in FY2022?") == {"VERIZON_2022_10K"}


def test_short_name_needs_a_standalone_word(doc_names):
    f = MetadataFilter(doc_names)
    assert f.identify_company("What was 3M's FY2018 net income?") == "3m"
    assert f.identify_company("How did AMD revenue change in FY22?") == "amd"


def test_multi_word_names_and_abbreviations(doc_names):
    f = MetadataFilter(doc_names)
    assert f.identify_company("What was American Express's largest liability?") == "americanexpress"
    assert f.identify_company("Is growth in JnJ's adjusted EPS expected to accelerate?") == "johnsonjohnson"


def test_finance_jargon_is_never_a_company(doc_names):
    # "EPS" is an in-order subsequence of PEPSICO; it must not select PepsiCo.
    f = MetadataFilter(doc_names)
    assert f.identify_company("What was the EPS growth last year?") is None


def test_renamed_company(doc_names):
    assert MetadataFilter(doc_names).identify_company("What was Square's revenue in 2020?") == "block"


def test_every_mentioned_year_is_kept(doc_names):
    f = MetadataFilter(doc_names)
    assert f.allowed_docs("Did Pfizer grow PP&E between FY20 and FY21?") == {"PFIZER_2020_10K", "PFIZER_2021_10K"}


def test_unknown_company_searches_everything(doc_names):
    assert MetadataFilter(doc_names).allowed_docs("What was Rivian's revenue in FY2022?") is None


def test_year_with_no_matching_filing_falls_back_to_all_company_filings(doc_names):
    f = MetadataFilter(doc_names)
    assert f.allowed_docs("What was Verizon's revenue in FY2015?") == {"VERIZON_2021_10K", "VERIZON_2022_10K"}


@pytest.mark.xfail(reason="Known bug (LIMITATIONS L-1): 'VaR' resolves to Activision as a subsequence",
                   strict=True)
def test_var_does_not_resolve_to_activision(doc_names):
    f = MetadataFilter(doc_names)
    assert f.identify_company("Looking at VaR, did the risk decrease in Q2 2023?") != "activisionblizzard"
