# Temporal Hallucination Framework (TrustRAG)

M.Tech research project on **temporal hallucination** in retrieval-augmented generation: answers
that were once true but are now outdated, because the retriever surfaced a stale source. The
domain is travel advisories (visa, vaccination and customs rules), where rules change and old
pages linger.

TrustRAG scores every retrieved fact for authority, freshness, agreement, evidence quality,
source reputation and conflict, picks one value per question slot, generates a cited answer,
and verifies each sentence before showing it.

- **Architecture, formulas and failure handling:** [docs/system_design.md](docs/system_design.md)
- **Benchmark datasheet:** [data/benchmark/DATASHEET.md](data/benchmark/DATASHEET.md)
- **Writing new blind test cases:** [docs/benchmark_authoring_protocol.md](docs/benchmark_authoring_protocol.md)
- **Latest results:** [results/REPORT.md](results/REPORT.md)

## Systems compared

| System | Description |
|---|---|
| S0 | Vanilla RAG: most similar passage, no trust scoring, no verification |
| S1 | Trust RAG with one fixed weight vector |
| S2 | Trust RAG with per-topic weights set by volatility (proposed method) |
| S3 | Trust RAG with per-topic weights learned from the train split |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make setup                 # install requirements
make test                  # unit tests
make demo                  # Streamlit UI on http://localhost:8501
```

In the UI, pick the **Benchmark corpus** knowledge base and a system, then ask e.g. *"How long
can Indian tourists stay in Thailand?"* and press **Compare RAG systems**.

## Reproducing the benchmark

```bash
make reproduce
```

This rebuilds everything from `data/benchmark/facts.yaml` in order: corpus and benchmark CSV,
FAISS index, query classifiers, benchmark checks, freshness ablation, reputation and S3 fitting
(train split only), the S0–S3 run on the test split, the report, the weight-sensitivity study
(train and dev only), and the unit tests. Individual steps are Makefile targets (`dataset`,
`index-benchmark`, `train-classifiers`, `fit`, `benchmark-experiments`, `report`, `sensitivity`).

## Repository layout

```text
trustrag/
  offline/        ingest, metadata, chunking + embedding (FAISS), change detection, reputation
  online/
    query/        intent/category classification, entities, clarification
    policy/       volatility score, retrieval strategy, weight vectors
    retrieval/    vector search, slot-value extraction, nationality-scope filter
    live/         live official-site fetch (GOV.UK, whitelist), Google fallback
    trust/        trust features, scoring, slot fusion
    generation/   prompt building, LLM backends (+ template fallback), verification
  policies/       authority tiers, categories, strategy rules, weight vectors
  pipeline.py     orchestration of S0-S3
  api.py, cli.py  FastAPI service and command line
ui/app.py         Streamlit demo
data/
  benchmark/      facts.yaml (source of truth), benchmark.csv, datasheet
  benchmark_corpus/  generated synthetic documents (75)
  corpus/         original 4-document demo corpus (used by unit tests)
  train/          classifier training queries, trust features for S3
experiments/      fitting, S0-S3 runs, metrics, statistics, sensitivity study, report
scripts/          build_dataset.py (benchmark generator and validator)
results/          main results, ablation, sensitivity figures, report
tests/            unit tests
```

## Configuration

- `trustrag/config.benchmark.yaml`: benchmark corpus and index, content-relative freshness (default for experiments and the UI).
- `trustrag/config.benchmark.legacy.yaml`: same, with the legacy crawl-date freshness rule (ablation baseline).
- `trustrag/config.yaml`: the 4-document demo corpus used by the unit tests.

LLM backends: `mock` (deterministic template writer, used for all benchmark runs), `ollama`
(llama3.1:8b, local), `openai` (gpt-4o-mini, `OPENAI_API_KEY`), `anthropic` (claude-haiku-5-5,
`ANTHROPIC_API_KEY`). If a backend is unavailable the answer falls back to the template writer.

The benchmark content is **synthetic**: modelled on real policy changes, but not verified against
live government pages. Do not use it as travel advice.
