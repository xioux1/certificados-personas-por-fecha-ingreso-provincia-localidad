from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def save_plots(run_dir: Path, model, x_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series) -> list[Path]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    cm_path = plots_dir / "confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(cm_path)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
        ax.set_title("ROC Curve")
        roc_path = plots_dir / "roc_curve.png"
        fig.savefig(roc_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(roc_path)

    return paths
