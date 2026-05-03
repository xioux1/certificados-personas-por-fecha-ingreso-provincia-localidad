from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _dict_to_table(data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            value_str = f"<pre>{html.escape(json.dumps(value, indent=2, ensure_ascii=False))}</pre>"
        else:
            value_str = html.escape(str(value))
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{value_str}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def generate_html_report(
    run_dir: Path,
    run_metadata: dict,
    dataset_summary: dict,
    config: dict,
    metrics: dict,
    plot_paths: list[Path],
    artifact_paths: list[Path],
    warnings: list[str] | None = None,
) -> Path:
    warnings = warnings or []

    image_blocks = []
    for p in plot_paths:
        rel = p.relative_to(run_dir)
        if p.exists():
            image_blocks.append(f'<div><h4>{html.escape(rel.name)}</h4><img src="{html.escape(str(rel))}" alt="{html.escape(rel.name)}"></div>')
        else:
            warnings.append(f"Missing plot: {rel}")

    report = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Experiment Report {html.escape(str(run_metadata.get('run_id')))}</title>
<style>body{{font-family:Arial,sans-serif;background:#f7f7f8;margin:0}}.container{{max-width:980px;margin:24px auto;background:white;padding:24px;border-radius:8px}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}}th{{width:240px;background:#f1f1f1}}img{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px}}</style>
</head><body><div class='container'>
<h1>Experiment Report</h1>
<h2>Run metadata</h2>{_dict_to_table(run_metadata)}
<h2>Dataset summary</h2>{_dict_to_table(dataset_summary)}
<h2>Model/config summary</h2>{_dict_to_table(config)}
<h2>Metrics</h2>{_dict_to_table(metrics)}
<h2>Plots</h2>{''.join(image_blocks) if image_blocks else '<p>No plots generated.</p>'}
<h2>Warnings / Notes</h2>{'<ul>' + ''.join(f'<li>{html.escape(w)}</li>' for w in warnings) + '</ul>' if warnings else '<p>None</p>'}
<h2>Artifacts</h2><ul>{''.join(f'<li>{html.escape(str(a.relative_to(run_dir)))}</li>' for a in artifact_paths)}</ul>
</div></body></html>
"""
    report_path = run_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path
