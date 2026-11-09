"""Load and validate static, policy, or learned trust-weight vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

POSITIVE_WEIGHT_KEYS = ("w_A", "w_F", "w_G", "w_E", "w_SR")
WEIGHT_KEYS = (*POSITIVE_WEIGHT_KEYS, "w_C")


def load_weights(config_path: str | Path, category: str, mode: str) -> tuple[dict[str, float], str]:
    config_path = Path(config_path)
    if mode == "learned":
        path = config_path.parent / "policies" / "weights_learned" / f"{category}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Learned weights are not available: {path}")
        key = None
    elif mode == "static":
        path, key = config_path.parent / "policies" / "weights_static.yaml", "default"
    elif mode == "policy":
        path, key = config_path.parent / "policies" / "weights_policy.yaml", category
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")

    values: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = (values.get("weights", values) if key is None else values.get(key))
    if not isinstance(raw, dict):
        raise ValueError(f"No {mode} weight vector configured for category {category}")
    if set(raw) != set(WEIGHT_KEYS):
        raise ValueError(f"Weight vector must contain exactly: {', '.join(WEIGHT_KEYS)}")
    weights = {name: float(raw[name]) for name in WEIGHT_KEYS}
    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative")
    if abs(sum(weights[name] for name in POSITIVE_WEIGHT_KEYS) - 1.0) > 1e-6:
        raise ValueError("Positive weights w_A, w_F, w_G, w_E, and w_SR must sum to 1")
    return weights, f"{path.name}:{key or category}"