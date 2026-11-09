"""Load calibrated query classifiers when trained model artifacts are available."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

try:
    import joblib
except ImportError:  # pragma: no cover - declared project dependency
    joblib = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback rules remain available
    SentenceTransformer = None


class CalibratedQueryModels:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        classifier_config = self.config.get("query_classifier", {})
        model_dir = Path(classifier_config.get("model_dir", "models/classifiers"))
        self.model_dir = model_dir if model_dir.is_absolute() else self.config_path.parent.parent / model_dir
        self.models: dict[str, Any] = {}
        self.encoder = None
        if joblib is None or SentenceTransformer is None:
            return
        for task in ("intent", "category"):
            model_path = self.model_dir / f"{task}.joblib"
            if model_path.exists():
                self.models[task] = joblib.load(model_path)
        if self.models:
            embedding = self.config["embedding"]
            try:
                self.encoder = SentenceTransformer(
                    embedding["model_name"],
                    device="cpu",
                    local_files_only=True,
                )
            except Exception:
                self.models.clear()

    def predict(self, query: str, task: str) -> list[tuple[str, float]] | None:
        model = self.models.get(task)
        if model is None or self.encoder is None:
            return None
        embedding = self.config["embedding"]
        vector = self.encoder.encode(
            [str(embedding.get("query_prefix", "")) + query],
            normalize_embeddings=bool(embedding.get("normalize", True)),
        )
        probabilities = model.predict_proba(vector)[0]
        ranked = sorted(
            ((str(label), float(probability)) for label, probability in zip(model.classes_, probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked