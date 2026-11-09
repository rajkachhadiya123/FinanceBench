"""Small pure helpers of retrieval and the API pipeline."""
from source.API.pipeline import is_not_found, public_chunk, split_header
from source.RETRIEVAL.retriever import RRF_K, Retriever


def test_rrf_rewards_agreement_between_lists():
    fused = Retriever._rrf_fuse([[1, 2, 3], [3, 1, 4]])
    assert fused[0] == 1        # ranks 1 and 2
    assert fused[1] == 3        # ranks 3 and 1
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_score_formula():
    # A single list: order is preserved, scores are 1 / (k + rank).
    assert Retriever._rrf_fuse([[7, 8, 9]]) == [7, 8, 9]
    assert RRF_K == 60


def test_split_header():
    header, text = split_header("[NIKE | 10-K | FY2023 | page 54]\nRevenues 51,217")
    assert header == "[NIKE | 10-K | FY2023 | page 54]"
    assert text == "Revenues 51,217"
    assert split_header("plain chunk text") == (None, "plain chunk text")


def test_public_chunk_hides_internal_fields():
    raw = {"rank": 1, "position": 99, "chunk_id": "x", "doc_name": "NIKE_2023_10K",
           "page_start": 54, "page_end": 55, "text": "[H]\nbody", "rerank_score": 0.9}
    assert public_chunk(raw) == {"rank": 1, "doc_name": "NIKE_2023_10K", "page_start": 54,
                                 "page_end": 55, "header": "[H]", "text": "body", "rerank_score": 0.9}


def test_not_found_detection():
    assert is_not_found("NOT FOUND")
    assert is_not_found("  not found. The excerpts lack it")
    assert not is_not_found("Revenue was $5 million; nothing was not found")
