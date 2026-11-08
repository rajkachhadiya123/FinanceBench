# experiments

Reserved for the **evaluation harness** — not built yet.

The plan: one YAML file per experiment describing a pipeline setup (embedding, reranker,
`top_k`, prompt, model, reasoning effort), run with a single command that answers the 150
questions, grades them (numeric auto-grader plus an LLM judge validated against
`data/PROCESSED/manual_verdicts.json`), stores the run and writes a report comparing it with
the 79.3% baseline.

Until then, experiments are run with the scripts in [tools/](../tools/README.md), and every
past experiment and its result is recorded in the
[experiment log](../documentation/experiment_log/README.md).
