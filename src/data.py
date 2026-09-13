"""Loading, cleaning, and validation for the Saudi employee attrition data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import ID_COLUMN, TARGET_COLUMN


@dataclass(frozen=True)
class PreparedData:
    """Clean dataset and an auditable summary of the cleaning steps."""

    frame: pd.DataFrame
    quality_report: pd.DataFrame


def _standardise_name(name: object) -> str:
    """Convert a source column name to a consistent snake_case name."""
    clean_name = str(name).replace("\xa0", " ").strip().lower()
    clean_name = re.sub(r"[^a-z0-9]+", "_", clean_name)
    return clean_name.strip("_")


def _standardise_text(value: object) -> object:
    """Remove invisible spaces from categorical values while retaining missing values."""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()
    return value


def load_and_prepare_data(source_path: Path) -> PreparedData:
    """Load the original Excel file and return validated, analysis-ready data.

    The function deliberately uses the original file rather than a pre-encoded
    version. This makes each cleaning and modelling decision reproducible.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    raw_frame = pd.read_excel(source_path)
    frame = raw_frame.rename(columns=_standardise_name).copy()
    frame = frame.map(_standardise_text)
    rows_before = len(frame)

    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError("The source file must include an Attrition column.")

    target_map = {"yes": 1, "no": 0}
    target_values = frame[TARGET_COLUMN].astype(str).str.lower().str.strip().map(target_map)
    invalid_target_count = int(target_values.isna().sum())
    frame = frame.loc[target_values.notna()].copy()
    frame[TARGET_COLUMN] = target_values.loc[target_values.notna()].astype(int)

    missing_before = int(frame.isna().sum().sum())
    categorical_columns = frame.select_dtypes(include="object").columns
    frame[categorical_columns] = frame[categorical_columns].fillna("Unknown")
    numeric_columns = frame.select_dtypes(include="number").columns.drop(
        TARGET_COLUMN, errors="ignore"
    )
    for column in numeric_columns:
        frame[column] = frame[column].fillna(frame[column].median())
    missing_after = int(frame.isna().sum().sum())

    if ID_COLUMN not in frame.columns:
        raise ValueError("The source file must include an ID column.")

    quality_report = pd.DataFrame(
        [
            {"check": "Rows loaded", "value": rows_before},
            {"check": "Duplicate rows removed", "value": duplicate_count},
            {"check": "Invalid attrition values removed", "value": invalid_target_count},
            {"check": "Missing values before treatment", "value": missing_before},
            {"check": "Missing values after treatment", "value": missing_after},
            {"check": "Final records", "value": len(frame)},
            {"check": "Final columns", "value": len(frame.columns)},
        ]
    )
    return PreparedData(frame=frame.reset_index(drop=True), quality_report=quality_report)


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent dashboard fields without altering the source variables.

    Age and salary are recorded as bands in the source. The derived fields make
    these bands easier to use as dashboard filters while preserving the source.
    """
    result = frame.copy()
    result["age_group"] = result["age"].astype(str)
    result["salary_band"] = result["monthlysalary"].astype(str)
    result["attrition_label"] = result[TARGET_COLUMN].map({1: "Left", 0: "Stayed"})
    return result


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return predictor columns and remove identifiers, target, and display labels."""
    # These dashboard fields duplicate the original age and salary variables, so
    # they are useful for filtering but should not be counted twice by a model.
    excluded = {ID_COLUMN, TARGET_COLUMN, "attrition_label", "age_group", "salary_band"}
    return frame.drop(columns=[column for column in excluded if column in frame.columns])
