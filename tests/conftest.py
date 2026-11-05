"""
Shared test fixtures.

The suite needs no API keys, no network and none of the gitignored data in
data/PROCESSED: external services are replaced by fakes and every database is a
temporary file. Only data/EVAL (the 150 benchmark questions, tracked in git) is read.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "EVAL" / "financebench_open_source.jsonl"

# A small slice of the real corpus naming patterns, enough to exercise the filter.
DOC_NAMES = [
    "3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q",
    "AMERICANEXPRESS_2022_10K", "AMD_2022_10K",
    "JOHNSON_JOHNSON_2022_10K", "JOHNSON_JOHNSON_2022Q4_EARNINGS", "JOHNSON_JOHNSON_2023Q2_EARNINGS",
    "PEPSICO_2022_10K", "PEPSICO_2023Q1_EARNINGS",
    "VERIZON_2021_10K", "VERIZON_2022_10K",
    "PFIZER_2020_10K", "PFIZER_2021_10K",
    "CVSHEALTH_2022_10K", "ACTIVISIONBLIZZARD_2019_10K", "JPMORGAN_2023Q2_10Q",
    "BLOCK_2020_10K", "MCDONALDS_8K_dated-2023-01-06",
]


@pytest.fixture(scope="session")
def benchmark_questions() -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def doc_names() -> list[str]:
    return list(DOC_NAMES)
