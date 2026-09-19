"""Shared filter definitions and pure filtering helpers for the dashboard."""

from __future__ import annotations

import pandas as pd


FILTER_DEFINITIONS = (
    ("department_filter", "Department", "department"),
    ("gender_filter", "Gender", "gender"),
    ("age_filter", "Age group", "age_group"),
    ("jobtitle_filter", "Job title", "jobtitle"),
    ("job_satisfaction_filter", "Job satisfaction", "job_satisfaction"),
    ("overtime_filter", "Overtime", "overtime"),
    ("experience_filter", "Years of experience", "years_experience"),
    ("salary_filter", "Salary band", "salary_band"),
)

FILTER_KEYS = tuple(key for key, _, _ in FILTER_DEFINITIONS)
FILTER_COLUMNS = tuple(column for _, _, column in FILTER_DEFINITIONS)

ORDERED_OPTIONS = {
    "job_satisfaction": ["Not satisfied", "Satisfied", "Very satisfied"],
    "overtime": ["No", "Yes"],
    "years_experience": [
        "Less than 5 years",
        "From 5 to 10 years",
        "From 11 to 15 years",
        "From 16 to 20 years",
        "From 21 to 25 years",
        "From 26 to 30 years",
        "From 31 to 35 years",
    ],
    "salary_band": [
        "Less than 5000 SAR",
        "From 5000 to 10000 S.R",
        "From 11000 to 15000 S.R",
        "From 16000 to 20000 S.R",
        "From 21000 to 25000 S.R",
        "From 26000 to 30000 S.R",
        "S.R 31000 - and more",
    ],
}


def apply_filters(frame: pd.DataFrame, selections: dict[str, list[object]]) -> pd.DataFrame:
    """Return records matching every non-empty multi-select filter."""
    filtered = frame.copy()
    for column, selected_values in selections.items():
        if selected_values and column in filtered.columns:
            filtered = filtered[filtered[column].isin(selected_values)]
    return filtered


def filter_options(frame: pd.DataFrame, column: str) -> list[object]:
    """Return stable, logically ordered non-missing options for a field."""
    values = frame[column].dropna().unique().tolist()
    if column in ORDERED_OPTIONS:
        preferred = [value for value in ORDERED_OPTIONS[column] if value in values]
        remaining = [value for value in values if value not in preferred]
        return preferred + sorted(remaining, key=lambda value: str(value).casefold())
    return sorted(values, key=lambda value: str(value).casefold())
