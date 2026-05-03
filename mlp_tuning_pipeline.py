"""Pipeline de referencia de clase aplicado al dataset del repositorio.

Etiquetas de trazabilidad:
- (extraido del notebook de clase)
- (ya estaba implementado, coincide con notebook)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "certificados-personas-por-fecha-ingreso-provincia-localidad.csv"
RANDOM_STATE = 42


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Carga y limpieza básica del dataset. (extraido del notebook de clase)"""
    df = pd.read_csv(path)
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    df["cantidad_certificados"] = pd.to_numeric(df["cantidad_certificados"], errors="coerce").fillna(0)
    df["cantidad_personas"] = pd.to_numeric(df["cantidad_personas"], errors="coerce").fillna(0)

    # Ingeniería de variables de fecha para MLP. (extraido del notebook de clase)
    df["anio"] = df["fecha_ingreso"].dt.year
    df["mes"] = df["fecha_ingreso"].dt.month
    df["dia"] = df["fecha_ingreso"].dt.day
    return df.dropna(subset=["anio", "mes", "dia"])


def make_preprocessor() -> ColumnTransformer:
    """Preprocesamiento para mezclar numéricas y categóricas. (extraido del notebook de clase)"""
    numeric_features = ["anio", "mes", "dia", "cantidad_certificados"]
    categorical_features = ["destino_provincia", "destino_localidad"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def classification_flow(df: pd.DataFrame) -> dict:
    """Clasificación: predice si hubo personas (>0). (extraido del notebook de clase)"""
    X = df[["anio", "mes", "dia", "destino_provincia", "destino_localidad", "cantidad_certificados"]]
    y = (df["cantidad_personas"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline_base = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(50,),
                    activation="relu",
                    alpha=0.0001,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline_base.fit(X_train, y_train)
    y_pred_base = pipeline_base.predict(X_test)

    scores_cv_base = cross_val_score(pipeline_base, X, y, cv=5, scoring="accuracy")

    param_grid = {
        "mlp__hidden_layer_sizes": [(20,), (50,), (100,), (50, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
    }
    _ = list(ParameterGrid(param_grid))

    grid_search = GridSearchCV(
        estimator=pipeline_base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X, y)

    param_dist = {
        "mlp__hidden_layer_sizes": [(20,), (50,), (100,), (150,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__alpha": np.logspace(-5, -1, 20),
        "mlp__learning_rate_init": np.logspace(-4, -1, 20),
    }
    random_search = RandomizedSearchCV(
        estimator=pipeline_base,
        param_distributions=param_dist,
        n_iter=15,
        scoring="accuracy",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )
    random_search.fit(X_train, y_train)

    return {
        "base_test_accuracy": accuracy_score(y_test, y_pred_base),
        "base_cv_mean": float(scores_cv_base.mean()),
        "grid_best_params": grid_search.best_params_,
        "grid_best_cv": float(grid_search.best_score_),
        "random_best_params": random_search.best_params_,
        "random_best_cv": float(random_search.best_score_),
    }


def regression_flow(df: pd.DataFrame) -> dict:
    """Regresión: predice cantidad_personas. (extraido del notebook de clase)"""
    X = df[["anio", "mes", "dia", "destino_provincia", "destino_localidad", "cantidad_certificados"]]
    y = df["cantidad_personas"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline_base = Pipeline(
        [
            ("preprocess", make_preprocessor()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(50,),
                    activation="relu",
                    alpha=0.0001,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline_base.fit(X_train, y_train)
    y_pred_base = pipeline_base.predict(X_test)

    scores_cv_base = cross_val_score(pipeline_base, X, y, cv=5, scoring="r2")

    param_grid = {
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001],
    }
    grid_search = GridSearchCV(
        estimator=pipeline_base,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X, y)

    param_dist = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (150,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__alpha": np.logspace(-5, -1, 20),
        "mlp__learning_rate_init": np.logspace(-4, -2, 20),
    }
    random_search = RandomizedSearchCV(
        estimator=pipeline_base,
        param_distributions=param_dist,
        n_iter=8,
        scoring="r2",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )
    random_search.fit(X_train, y_train)

    return {
        "base_mae": float(mean_absolute_error(y_test, y_pred_base)),
        "base_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_base))),
        "base_r2": float(r2_score(y_test, y_pred_base)),
        "base_cv_mean_r2": float(scores_cv_base.mean()),
        "grid_best_params": grid_search.best_params_,
        "grid_best_cv_r2": float(grid_search.best_score_),
        "random_best_params": random_search.best_params_,
        "random_best_cv_r2": float(random_search.best_score_),
    }


if __name__ == "__main__":
    data = load_dataset()
    print("=== Clasificación ===")
    print(classification_flow(data))
    print("=== Regresión ===")
    print(regression_flow(data))
