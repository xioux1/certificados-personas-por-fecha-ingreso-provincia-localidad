from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class TrainArtifacts:
    model: object
    x_test: pd.DataFrame
    y_test: pd.Series
    y_pred: pd.Series
    train_summary: dict


def train_model(pipeline, x: pd.DataFrame, y: pd.Series, train_config: dict) -> TrainArtifacts:
    test_size = float(train_config.get("test_size", 0.2))
    random_state = int(train_config.get("random_state", 42))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return TrainArtifacts(
        model=pipeline,
        x_test=x_test,
        y_test=y_test,
        y_pred=pd.Series(y_pred, index=y_test.index),
        train_summary={"train_rows": len(x_train), "test_rows": len(x_test), "test_size": test_size},
    )
