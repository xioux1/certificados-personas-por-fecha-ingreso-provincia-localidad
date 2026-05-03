from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a dictionary")
    return config


def save_config(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
