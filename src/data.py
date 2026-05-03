from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    df["cantidad_certificados"] = pd.to_numeric(df["cantidad_certificados"], errors="coerce").fillna(0)
    df["cantidad_personas"] = pd.to_numeric(df["cantidad_personas"], errors="coerce").fillna(0)
    df["anio"] = df["fecha_ingreso"].dt.year
    df["mes"] = df["fecha_ingreso"].dt.month
    df["dia"] = df["fecha_ingreso"].dt.day
    return df.dropna(subset=["anio", "mes", "dia"]).copy()


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_min": str(df["fecha_ingreso"].min()),
        "date_max": str(df["fecha_ingreso"].max()),
        "province_count": int(df["destino_provincia"].nunique()),
        "locality_count": int(df["destino_localidad"].nunique()),
    }
