# Graph Report - FinanceBench  (2026-11-08)

## Corpus Check
- Corpus is ~36,842 words - fits in a single context window. You may not need a graph.

## Summary
- 668 nodes · 1394 edges · 31 communities (26 shown, 5 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 149 edges (avg confidence: 0.89)
- Token cost: 157,813 input · 0 output

## Community Hubs (Navigation)
- Hybrid Retriever and Evaluators
- API App and Pipeline
- Index Builders and Embeddings
- Answer Evaluation Scripts
- Request Tracing
- Chunking and Context Headers
- API Integration Tests
- Answer Cache and Rate Limiter
- Metadata Filter
- LLM Generation and Reranker Client
- Guardrail Checks
- Catalogue, Costs and Embedding Cache
- Early PDF Analysis Scripts
- README, Tests and Docker Docs
- Experiment Log and Prompt History
- Source Stage Overview
- Docker Compose Service
- Evaluation Results and Harness Plan
- Engineering Steps and Limits
- Evaluation Rules and Results
- Failure Analysis and Filter Bug
- Project Context Files
- Final Pipeline and Reranking
- Embeddings, Tracing and Costs Docs
- User Rules and API Limits
- Analysis Folder Docs
- Local Files Limit
- Charts Limit
- Environment Gotchas

## God Nodes (most connected - your core abstractions)
1. `Retriever` - 38 edges
2. `span()` - 28 edges
3. `query()` - 22 edges
4. `MetadataFilter` - 20 edges
5. `main()` - 20 edges
6. `retrieve()` - 18 edges
7. `QAPipeline` - 18 edges
8. `AnswerGenerator` - 16 edges
9. `check_output()` - 15 edges
10. `rebuild_chunks()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Step 3: Chunk context headers (85% @5)` --semantically_similar_to--> `Headed Chunks (context header prefix)`  [INFERRED] [semantically similar]
  documentation/experiment_log/README.md → HANDOFF.md
- `L-6 LLM Picks Wrong Figures in Calculations` --semantically_similar_to--> `Rejected Experiments (BGE, cross-encoders, prompt v3, v4+high)`  [INFERRED] [semantically similar]
  documentation/LIMITATIONS.md → HANDOFF.md
- `L-8 Rate Limits Shape the System (Cerebras)` --semantically_similar_to--> `API Keys and Rate Limits (Cerebras, Cohere)`  [INFERRED] [semantically similar]
  documentation/LIMITATIONS.md → HANDOFF.md
- `test_var_does_not_resolve_to_activision (xfail strict)` --conceptually_related_to--> `VaR -> Activision Metadata Filter Bug`  [INFERRED]
  tests/README.md → HANDOFF.md
- `Step 3: Chunk context headers (85% @5)` --semantically_similar_to--> `Chunk context header`  [INFERRED] [semantically similar]
  documentation/experiment_log/README.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Final retrieval and answer pipeline stages** — handoff_metadata_filter, handoff_qwen3_embedding, handoff_bm25, handoff_rrf_merge, handoff_cohere_rerank, handoff_cerebras_gpt_oss_120b, handoff_not_found_retry [EXTRACTED 1.00]
- **Retrieval causes of ~14 remaining failures** — documentation_limitations_l1_single_fiscal_year, documentation_limitations_l2_multi_statement, documentation_limitations_l3_fixed_chunks, documentation_limitations_failure_breakdown [EXTRACTED 1.00]
- **Per-process state that blocks multi-replica scaling** — documentation_limitations_l14_one_process, handoff_sqlite_answer_cache, handoff_embedding_cache, handoff_rate_limiter [EXTRACTED 1.00]
- **FilingLens query pipeline (filter, hybrid search, RRF, rerank, LLM)** — documentation_experiment_log_readme_metadata_filter, documentation_experiment_log_readme_bm25_search, documentation_experiment_log_readme_faiss_index, documentation_experiment_log_readme_rrf_fusion, documentation_experiment_log_readme_cohere_rerank_step, documentation_experiment_log_readme_prompt_v2 [INFERRED 0.85]

## Communities (31 total, 5 thin omitted)

### Community 0 - "Hybrid Retriever and Evaluators"
Cohesion: 0.07
Nodes (21): CohereReranker, EmbeddingCache, main(), Retriever, test_embedding_cache_is_lru(), test_rrf_rewards_agreement_between_lists(), test_rrf_score_formula(), main() (+13 more)

### Community 1 - "API App and Pipeline"
Cohesion: 0.06
Nodes (29): Chunk, clear_cache(), documents(), get_trace(), guard_request(), Guardrails, health(), HealthResponse (+21 more)

### Community 2 - "Index Builders and Embeddings"
Cohesion: 0.06
Nodes (18): load_chunk_texts(), main(), _count_truncated(), _embed(), _finished_shards(), iter_chunk_texts(), main(), _report() (+10 more)

### Community 3 - "Answer Evaluation Scripts"
Cohesion: 0.07
Nodes (29): answer_hash(), final_answers(), main(), mcnemar(), outcome(), answer_numbers(), strict_in_text(), answer_in_text() (+21 more)

### Community 4 - "Request Tracing"
Cohesion: 0.07
Nodes (18): _NoSpan, _percentile(), Span, span(), start_trace(), _to_db(), Trace, TraceStore (+10 more)

### Community 5 - "Chunking and Context Headers"
Cohesion: 0.10
Nodes (19): build_full_text_with_page_offsets(), chunk_document(), extract_pages(), main(), offset_to_page(), sliding_window_chunks(), _events(), test_detects_items_statements_notes_and_parent_company() (+11 more)

### Community 6 - "API Integration Tests"
Cohesion: 0.08
Nodes (14): GenerationFailed, DailyLimitReached, ask(), client(), FakePipeline, test_clear_cache(), test_cost_is_estimated_per_request(), test_injection_is_blocked_before_the_pipeline() (+6 more)

### Community 7 - "Answer Cache and Rate Limiter"
Cohesion: 0.13
Nodes (9): RateLimiter, AnswerCache, make_key(), normalize(), test_cache_round_trip_and_stats(), test_cache_survives_reopen(), test_normalize_and_key(), test_rate_limit_per_client_and_group() (+1 more)

### Community 8 - "Metadata Filter"
Cohesion: 0.14
Nodes (13): MetadataFilter, normalize(), parse_doc_name(), test_every_mentioned_year_is_kept(), test_finance_jargon_is_never_a_company(), test_full_name_and_year_select_one_filing(), test_multi_word_names_and_abbreviations(), test_parse_doc_name_regular_and_irregular() (+5 more)

### Community 9 - "LLM Generation and Reranker Client"
Cohesion: 0.12
Nodes (5): AnswerGenerator, check_formula(), test_build_context_labels_each_chunk(), test_check_formula(), test_final_answer_keeps_multi_line_lists()

### Community 10 - "Guardrail Checks"
Cohesion: 0.14
Nodes (17): check_input(), check_output(), _decimals(), extract_figures(), _found(), needs_disclaimer(), test_attacks_are_blocked(), test_disclaimer_only_for_advice_requests() (+9 more)

### Community 11 - "Catalogue, Costs and Embedding Cache"
Cohesion: 0.13
Nodes (8): catalogue(), estimate_cost(), _price(), _CountingEmbedder, _FakeIndex, test_catalogue_groups_and_normalizes_filings(), test_cost_estimate(), test_dense_search_embeds_each_text_once()

### Community 12 - "Early PDF Analysis Scripts"
Cohesion: 0.20
Nodes (9): containment_ratio(), load_evidence_samples(), main(), normalize(), safe(), check_for_images_and_charts(), safe(), show_plain_text() (+1 more)

### Community 13 - "README, Tests and Docker Docs"
Cohesion: 0.12
Nodes (16): Docker deployment guide (filinglens-api container), filinglens-state volume (cache.db, traces.db), data/PROCESSED read-only mount, Docker troubleshooting table, REST API endpoints (/query, /retrieve, /documents, /health, /traces, /stats, /cache), Environment configuration (.env variables), Project layout, FilingLens root README (+8 more)

### Community 14 - "Experiment Log and Prompt History"
Cohesion: 0.13
Nodes (17): BM25 keyword search (bm25s), Experiment Log: Retrieval & Embedding Evolution, FAISS IndexFlatIP exact search, How we measure (split, answer-in-top-k, strict matching, McNemar), Rule-based metadata filter (company + fiscal year), MiniLM silent 256-token truncation, Where missed answers rank, Prompt history (v1-v4) (+9 more)

### Community 15 - "Source Stage Overview"
Cohesion: 0.18
Nodes (15): Step 3: Chunk context headers (85% @5), Answer cache (SQLite, version-aware keys, no expiry), Chunk context header, Query-embedding LRU cache, Guardrails feature, Request flow (guardrails, cache, filter, hybrid search, rerank, LLM, output checks), Per-client rate limiting (10 /query, 30 /retrieve per min), API (QAPipeline, FastAPI app) (+7 more)

### Community 16 - "Docker Compose Service"
Cohesion: 0.13
Nodes (14): build, container_name, env_file, environment, image, ports, restart, volumes (+6 more)

### Community 17 - "Evaluation Results and Harness Plan"
Cohesion: 0.16
Nodes (14): Step 2: Cohere rerank-v4.0-pro (82% @5, p<0.001), End-to-end results, all 150 questions, Grading rules (CORRECT/PARTIAL/WRONG/NOT FOUND), Idea: LLM judge for free-text grading, L-16 No Evaluation Harness, L-9 Free-Text Answers Graded by Hand, Evaluation harness (planned: YAML configs, LLM judge, run registry), manual_verdicts.json (hand grading) (+6 more)

### Community 18 - "Engineering Steps and Limits"
Cohesion: 0.16
Nodes (14): Project Commands (pytest, uvicorn, docker, test_answers), L-13 Guardrails Are Rule-Based, L-14 One API Process, Next: AI Agent Using the API as a Tool, Docker Deployment (python:3.10-slim, CPU torch, Qwen baked in), GET /documents Catalogue, Engineering Steps (docs, API, guardrails, tracing, tests, Docker), FastAPI Service (/query, /retrieve, /health, /cache) (+6 more)

### Community 19 - "Evaluation Rules and Results"
Cohesion: 0.20
Nodes (10): L-10 Baseline Comparison Not a Clean Run, L-11 Small Samples, L-19 Tables Are Linear Text (PyMuPDF), L-3 Fixed-Size Chunks Cut Tables, L-4 Section Labels Are Heuristic, BM25 Sparse Search, Headed Chunks (context header prefix), Paired McNemar Significance Test (+2 more)

### Community 20 - "Failure Analysis and Filter Bug"
Cohesion: 0.27
Nodes (10): Where the 31 remaining failures come from, Idea: multi-year metadata filter, Idea: whole-statement retrieval, Remaining 31 Failures Breakdown, L-12 Some Expected Answers Are Debatable, L-1 Metadata Filter Keeps One Fiscal Year, L-2 Calculations Need Several Statements, L-7 Answers Vary Between Runs (+2 more)

### Community 21 - "Project Context Files"
Cohesion: 0.28
Nodes (9): CLAUDE.md - FilingLens notes for Claude, graphify Knowledge Graph Usage, The metric bug (lenient matcher), Known Limitations, documentation folder index, Early retrieval PDF reports (optimistic, pre-strict matcher), HANDOFF - full project notes, How a New Chat Finds Things (+1 more)

### Community 22 - "Final Pipeline and Reranking"
Cohesion: 0.29
Nodes (8): Local cross-encoder reranker experiments, L-5 External API Limits (Cohere trial), Cohere rerank-v4.0-pro, FilingLens RAG System, Final Pipeline (filter, hybrid retrieval, rerank, LLM, retry), FinanceBench Benchmark (150 questions, 368 filings), NOT FOUND Retry with Top 10, RRF Merge (k=60)

### Community 23 - "Embeddings, Tracing and Costs Docs"
Cohesion: 0.25
Nodes (7): L-15 Cost Figures Are Estimates, Google Colab T4 Embedding Runs, Cost Estimates (costs.py, cost_usd), Query-Embedding LRU Cache (EmbeddingCache, 1,024), Qwen3-Embedding-0.6B Dense Search, Tracing (traces + spans, traces.db, 30-day retention), colab_embed_qwen.py (Colab T4, resumable)

### Community 24 - "User Rules and API Limits"
Cohesion: 0.29
Nodes (4): L-6 LLM Picks Wrong Figures in Calculations, L-8 Rate Limits Shape the System (Cerebras), API Keys and Rate Limits (Cerebras, Cohere), Cerebras gpt-oss-120b (prompt v2, reasoning medium)

### Community 25 - "Analysis Folder Docs"
Cohesion: 0.50
Nodes (4): analysis folder README, pdf_extraction_test.py (evidence containment check), table_extraction_demo.py (PyMuPDF vs pdfplumber), Pinned runtime dependencies (fastapi, sentence-transformers, torch, faiss-cpu, bm25s, httpx, pymupdf)

## Knowledge Gaps
- **37 isolated node(s):** `FinanceBench Benchmark (150 questions, 368 filings)`, `RRF Merge (k=60)`, `GET /documents Catalogue`, `Docker Deployment (python:3.10-slim, CPU torch, Qwen baked in)`, `Environment Gotchas (Colab, Docker PATH, scratchpad)` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 219 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `tools folder README (evaluation + data prep)` connect `Evaluation Results and Harness Plan` to `Hybrid Retriever and Evaluators`, `Chunking and Context Headers`, `README, Tests and Docker Docs`, `Project Context Files`, `Embeddings, Tracing and Costs Docs`?**
  _High betweenness centrality (0.187) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `Retriever` (e.g. with `QAPipeline` and `CohereReranker`) actually correct?**
  _`Retriever` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `FinanceBench Benchmark (150 questions, 368 filings)`, `RRF Merge (k=60)`, `GET /documents Catalogue` to the rest of the system?**
  _37 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Hybrid Retriever and Evaluators` be split into smaller, more focused modules?**
  _Cohesion score 0.06734867860187553 - nodes in this community are weakly interconnected._
- **Why does `FilingLens root README` connect `README, Tests and Docker Docs` to `Evaluation Results and Harness Plan`, `Project Context Files`, `Experiment Log and Prompt History`, `Source Stage Overview`?**
  _High betweenness centrality (0.125) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `query()` (e.g. with `GenerationFailed` and `QAPipeline`) actually correct?**
  _`query()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Should `API App and Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.06246799795186892 - nodes in this community are weakly interconnected._