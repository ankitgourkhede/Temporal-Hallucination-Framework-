# TrustRAG Evaluation Report

This report was generated from the saved experiment outputs.

## Main Results

| system | accuracy | abstention_rate | selective_accuracy | accuracy_ci_low | accuracy_ci_high | hallucination_rate | faithfulness | latency_p50_ms     | latency_p95_ms     | retry_rate | ece      | trust_auroc |
| ------ | -------- | --------------- | ------------------ | --------------- | ---------------- | ------------------ | ------------ | ------------------ | ------------------ | ---------- | -------- | ----------- |
| S0     | 0.0      | 0.0             | 0.0                | 0.0             | 0.0              | 0.0                |              | 430.19654899944726 | 430.19654899944726 | 0.0        | 0.899351 |             |
| S1     | 1.0      | 0.0             | 1.0                | 1.0             | 1.0              | 0.0                | 1.0          | 386.7531850000887  | 386.7531850000887  | 0.0        | 0.958013 |             |
| S2     | 1.0      | 0.0             | 1.0                | 1.0             | 1.0              | 0.0                | 1.0          | 248.95742500029883 | 248.95742500029883 | 0.0        | 0.955917 |             |

## Statistical Tests

| comparison        | b | c | method | statistic | p_value | holm_adjusted_p |
| ----------------- | - | - | ------ | --------- | ------- | --------------- |
| S0_vs_S2_accuracy | 0 | 1 | exact  | 0.0       | 1.0     | 1.0             |

## Run Manifest

```json
{
  "git_hash": "1a7a278cc312c835db9a5872b56bc4769202ea63",
  "config_sha256": "8db28dd8e1bd31aeea47fcb9eddf20586feb56114958c176ac9d0c76a440901d",
  "model": "BAAI/bge-small-en-v1.5",
  "llm": "mock",
  "seed": 42,
  "as_of_date": "2026-11-09",
  "benchmark_mode": true,
  "smoke": true
}
```

**This is a smoke run, not thesis benchmark evidence.**
