from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .features import make_preprocessor


def build_model(model_config: dict) -> Pipeline:
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(model_config.get("hidden_layer_sizes", [50])),
        activation=model_config.get("activation", "relu"),
        alpha=float(model_config.get("alpha", 0.0001)),
        max_iter=int(model_config.get("max_iter", 500)),
        random_state=int(model_config.get("random_state", 42)),
    )
    return Pipeline([("preprocess", make_preprocessor()), ("model", mlp)])
