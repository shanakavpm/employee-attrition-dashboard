"""Portable dashboard results generated offline by the training pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TABLES = ("frame", "quality_report", "metrics", "risk_review", "importance", "gender_fairness", "age_fairness")


def save_dashboard_data(path: Path, *, best_model_name: str, **tables: pd.DataFrame) -> None:
    """Write one complete bundle so the hosted app never mixes model runs."""
    payload = {"schema_version": 1, "best_model_name": best_model_name}
    for name in TABLES:
        payload[name] = json.loads(tables[name].to_json(orient="split", index=False, double_precision=15))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def load_dashboard_data(path: Path) -> dict:
    """Read results without importing or executing model training code."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported dashboard results version")
    result = {"best_model_name": payload["best_model_name"]}
    for name in TABLES:
        table = payload[name]
        result[name] = pd.DataFrame(table["data"], columns=table["columns"])
    return result
