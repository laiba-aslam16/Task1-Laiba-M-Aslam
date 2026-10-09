"""Step 3: find and neutralise outliers with the IQR rule.

A value is an outlier when it is below Q1 - k*IQR or above Q3 + k*IQR (k = 1.5).
Outliers are capped at the fence (winsorised): the car stays in the data, but the
extreme value stops dominating the model. Right-skewed columns use fences on a log scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def iqr_fences(values: pd.Series, multiplier: float, log_scale: bool) -> dict:
    """Quartiles, IQR and fences, all in the original units."""
    clean = values.dropna()
    x = np.log1p(clean) if log_scale else clean
    q1_t, q3_t = x.quantile(0.25), x.quantile(0.75)
    spread = q3_t - q1_t
    lower_t, upper_t = q1_t - multiplier * spread, q3_t + multiplier * spread

    back = np.expm1 if log_scale else (lambda v: v)
    q1, q3 = float(back(q1_t)), float(back(q3_t))
    return {
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "lower": float(back(lower_t)),
        "upper": float(back(upper_t)),
    }


def _describe(df: pd.DataFrame, column: str, scale: str, multiplier: float, only_rows: str | None) -> dict:
    values = df.query(only_rows)[column] if only_rows else df[column]
    stats = iqr_fences(values, multiplier, log_scale=(scale == "log"))
    stats["n_low"] = int((values < stats["lower"]).sum())
    stats["n_high"] = int((values > stats["upper"]).sum())
    return stats


def _report_row(column: str, scale: str, stats: dict, action: str, flagged_ok: bool = True) -> dict:
    return {
        "Column": column,
        "Scale": scale,
        "Q1": stats["q1"],
        "Q3": stats["q3"],
        "IQR": stats["iqr"],
        "Lower fence": stats["lower"],
        "Upper fence": stats["upper"],
        "Flagged low": stats["n_low"] if flagged_ok else np.nan,
        "Flagged high": stats["n_high"] if flagged_ok else np.nan,
        "Action": action,
    }


def treat_outliers(
    df: pd.DataFrame,
    columns: dict,
    excluded: dict,
    multiplier: float,
    treat_target: bool,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap outliers column by column. Returns the treated table and an outlier summary table."""
    out = df.copy()
    rows = []

    for column, scale in columns.items():
        stats = _describe(out, column, scale, multiplier, only_rows=None)
        if stats["iqr"] == 0:
            # Most values sit exactly on one number (e.g. 100%), so the fences would
            # wrongly flag every value below it. Capping is skipped in that case.
            action = "skipped: IQR is 0 (most values sit on one number)"
            rows.append(_report_row(column, scale, stats, action, flagged_ok=False))
            continue
        if column == target and not treat_target:
            action = "flagged only (TREAT_TARGET_OUTLIERS is False)"
        else:
            out[column] = out[column].clip(lower=stats["lower"], upper=stats["upper"])
            action = "capped to fences"
        rows.append(_report_row(column, scale, stats, action))

    for column, (reason, only_rows) in excluded.items():
        stats = _describe(out, column, "raw", multiplier, only_rows=only_rows)
        rows.append(_report_row(column, "raw", stats, f"not capped: {reason}"))

    return out, pd.DataFrame(rows)
