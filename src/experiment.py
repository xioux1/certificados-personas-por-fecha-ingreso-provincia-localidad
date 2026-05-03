from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import load_config, save_config
from .data import dataset_summary, load_dataset
from .evaluate import evaluate_predictions
from .features import build_features
from .models import build_model
from .plots import save_plots
from .reporting import generate_html_report
from .train import train_model


def run_experiment(config_path: Path) -> Path:
    config = load_config(config_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get("results_dir", "results")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config.yaml")

    df = load_dataset(Path(config["dataset_path"]))
    ds_summary = dataset_summary(df)
    x, y = build_features(df)

    model = build_model(config.get("model", {}))
    trained = train_model(model, x, y, config.get("train", {}))

    y_score = trained.model.predict_proba(trained.x_test)[:, 1] if hasattr(trained.model, "predict_proba") else None
    metrics = evaluate_predictions(trained.y_test, trained.y_pred, y_score=y_score)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"y_true": trained.y_test, "y_pred": trained.y_pred})
    pred_path = run_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    plot_paths = save_plots(run_dir, trained.model, trained.x_test, trained.y_test, trained.y_pred)

    run_metadata = {
        "run_id": run_id,
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "training_summary": trained.train_summary,
    }
    artifact_paths = [run_dir / "config.yaml", metrics_path, pred_path, *plot_paths]
    generate_html_report(
        run_dir=run_dir,
        run_metadata=run_metadata,
        dataset_summary=ds_summary,
        config=config,
        metrics=metrics,
        plot_paths=plot_paths,
        artifact_paths=artifact_paths,
    )
    return run_dir
