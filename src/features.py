from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = ["anio", "mes", "dia", "destino_provincia", "destino_localidad", "cantidad_certificados"]
TARGET_COLUMN = "cantidad_personas"


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN] > 0).astype(int)
    return x, y


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["anio", "mes", "dia", "cantidad_certificados"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["destino_provincia", "destino_localidad"]),
        ]
    )
