# Temporal Hallucination Framework

A research and engineering workspace for studying, detecting, and mitigating temporal hallucinations in LLM-driven systems.

## Project goal

This repository is structured as a modular research project with separate areas for:

- literature review and source material
- datasets and experiment tracking
- notebooks for analysis and prototyping
- Python package modules for the core pipeline
- evaluation, reporting, and presentation assets

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m temporal_hallucination
```

## Repository structure

```text
Temporal-Hallucination-Framework/
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── handbook/
│   ├── Chapter-01-Introduction.md
│   ├── Chapter-02-LLM-Fundamentals.md
│   ├── Chapter-03-Hallucinations.md
│   ├── Chapter-04-Information-Retrieval.md
│   ├── Chapter-05-RAG.md
│   ├── Chapter-06-Multi-Source-Retrieval.md
│   ├── Chapter-07-Fact-Classification.md
│   ├── Chapter-08-Volatility-Classification.md
│   ├── Chapter-09-Source-Freshness.md
│   ├── Chapter-10-Reconciliation.md
│   ├── Chapter-11-Adaptive-Weighting.md
│   ├── Chapter-12-Temporal-Hallucination.md
│   ├── Chapter-13-Evaluation.md
│   ├── Chapter-14-Deployment.md
│   └── References.md
├── literature/
│   ├── papers/
│   ├── summaries/
│   ├── ieee_notes/
│   └── bibliography.bib
├── thesis/
│   ├── proposal/
│   ├── chapters/
│   ├── figures/
│   ├── tables/
│   ├── references/
│   └── final_submission/
├── patent/
│   ├── novelty_notes.md
│   ├── claims.md
│   └── diagrams/
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── benchmark/
│   ├── labelled/
│   └── evaluation/
├── experiments/
│   ├── exp001/
│   ├── exp002/
│   ├── exp003/
│   └── logs/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── volatility_classifier.ipynb
│   ├── embeddings.ipynb
│   └── evaluation.ipynb
├── src/
│   └── temporal_hallucination/
│       ├── adapters/
│       ├── retrieval/
│       ├── extraction/
│       ├── preprocessing/
│       ├── classifiers/
│       ├── freshness/
│       ├── reconciliation/
│       ├── adaptive_weighting/
│       ├── temporal_detector/
│       ├── llm/
│       ├── api/
│       ├── ui/
│       ├── utils/
│       ├── config/
│       ├── __init__.py
│       ├── __main__.py
│       └── pipeline.py
├── models/
│   ├── embeddings/
│   ├── classifiers/
│   ├── checkpoints/
│   └── exported/
├── vector_db/
├── cache/
├── evaluation/
│   ├── metrics/
│   ├── baselines/
│   ├── calibration/
│   ├── reports/
│   └── plots/
├── tests/
├── scripts/
├── docs/
├── presentations/
│   ├── proposal/
│   ├── progress_review/
│   ├── final_viva/
│   └── ieee_presentation/
└── .venv/  # local environment (not tracked)
```

## Core Python package layout

The code is organized around a modular pipeline:

- `src/temporal_hallucination/config` – settings and environment configuration
- `src/temporal_hallucination/preprocessing` – cleaning and normalization
- `src/temporal_hallucination/retrieval` – document and evidence retrieval
- `src/temporal_hallucination/extraction` – claim and fact extraction
- `src/temporal_hallucination/freshness` – temporal recency and source checks
- `src/temporal_hallucination/classifiers` – fact and volatility models
- `src/temporal_hallucination/reconciliation` – conflict resolution logic
- `src/temporal_hallucination/adaptive_weighting` – weighting signals by confidence and time sensitivity
- `src/temporal_hallucination/temporal_detector` – core detection logic
- `src/temporal_hallucination/api` – service interface for endpoint integration
- `src/temporal_hallucination/ui` – optional dashboards or front-end wrappers
- `src/temporal_hallucination/utils` – reusable helper functions

## Recommended next steps

1. Create the research dataset schema.
2. Implement the retrieval and preprocessing pipeline.
3. Add a benchmark evaluation runner.
4. Build a small FastAPI endpoint for demonstration.
5. Track experiments under `experiments/expNNN`.

## Notes

This scaffold is intentionally lightweight and intentionally designed to grow into a full research codebase without forcing a rigid architecture too early.
