I made my embedding model 27× bigger. Retrieval improved by 8 questions out of 71.
Then I added a reranker. It improved by 15 more, in one step.

For the last few weeks I've been building FilingLens, a question-answering system over SEC filings, tested on FinanceBench: 150 analyst questions over 368 annual, quarterly and earnings reports.

The hard part isn't the answering. It's finding the right page. A company can have around 9 near-identical filings, and one balance sheet looks a lot like the next.

What I built: a filter that reads the company and year from the question, hybrid search (Qwen3 embeddings + BM25), a Cohere reranker, and gpt-oss-120b to answer with citations.

The numbers:
• Answer passage in the top 5: 49% → 85%
• Reranker alone: +16 / −1 questions (p < 0.001)
• End-to-end: 119/150 correct (79.3%)

For context, the FinanceBench paper (2023) reported about 19% for GPT-4-Turbo with a basic shared vector store. A newer model plus careful retrieval gets a lot further.

What didn't work: in 16 of my 31 remaining wrong answers, every gold page was already in front of the model. Two prompt rewrites fixed 0 of 18 calculation errors. More rules didn't help, so next I'm giving the model tools, starting with a calculator.

Code, experiment log and every measurement: [REPO_LINK]

#RAG #MachineLearning #LLM

---

<!-- Alternative hooks (not part of the post) -->
<!-- A: "In half of my AI's wrong answers, the right page was already in front of it." -->
<!-- B: "My first evaluation said 74%. The real number was 56%. The bug was in my grader, not my model." -->
