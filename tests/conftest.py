"""Keep unit tests independent of locally trained, git-ignored artifacts.

`make train-classifiers` writes models/classifiers/*.joblib and `make fit` writes
trustrag/policies/weights_learned/*.yaml. Tests assert the documented behaviour of a
fresh checkout (keyword routing, S3 unavailable until fitted), so both are hidden here.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_checkout_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    from trustrag.online.policy import weights as weights_module
    from trustrag.online.query import classifier as classifier_module

    monkeypatch.setattr(classifier_module.CalibratedQueryModels, "predict", lambda self, query, task: None)

    original = weights_module.load_weights

    def without_learned(config_path, category, mode):
        if mode == "learned":
            raise FileNotFoundError("Learned weights are hidden in unit tests")
        return original(config_path, category, mode)

    monkeypatch.setattr(weights_module, "load_weights", without_learned)
    monkeypatch.setattr("trustrag.pipeline.load_weights", without_learned)
