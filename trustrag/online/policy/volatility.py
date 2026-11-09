"""Temporal Volatility Score computation from observed primary-source changes."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def compute_tvs(events: list[dict[str, Any]], expert_prior: float, as_of: str | date) -> dict[str, Any]:
    as_of_date = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    parsed = []
    for event in events:
        detected = event.get("detected_at")
        if not detected:
            continue
        try:
            detected_date = datetime.fromisoformat(str(detected)[:10]).date()
        except ValueError:
            continue
        parsed.append((detected_date, float(event.get("diff_magnitude", 0.0))))

    observed_days = (as_of_date - min((item[0] for item in parsed), default=as_of_date)).days
    if len(parsed) < 3 or observed_days < 180:
        return {"tvs": float(expert_prior), "tvs_source": "expert", "changes_per_year": None, "mean_diff": None}

    changes_per_year = len(parsed) * 365.0 / observed_days
    mean_diff = sum(item[1] for item in parsed) / len(parsed)
    update_frequency = min(changes_per_year / 4.0, 1.0)
    tvs = 0.5 * update_frequency + 0.3 * mean_diff + 0.2 * expert_prior
    return {
        "tvs": min(1.0, max(0.0, tvs)),
        "tvs_source": "change_history",
        "changes_per_year": changes_per_year,
        "mean_diff": mean_diff,
    }