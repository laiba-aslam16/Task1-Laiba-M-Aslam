"""Step 4: engineer new predictive features and encode categories for machine learning."""
from __future__ import annotations

import pandas as pd

from src import config as cfg

# Shown in the report. Name -> plain-English definition.
ENGINEERED_FEATURES = {
    "car_age": "listing year minus model year (never below 0)",
    "km_per_year": "mileage divided by car age (age counted as at least 1 year)",
    "inspection_score": "average of the five inspection percentages",
    "engine_cc": "engine size read from the car name; electric cars = 0; gaps filled by KNN",
    "is_premium_brand": "1 if the brand is on the premium list, else 0",
    "is_automatic": "1 if the gearbox is Automatic, else 0",
}

# Numeric columns kept in the final model table (identifiers, target and dummies are added separately).
MODEL_NUMERIC = [
    "model_year",
    "car_age",
    "mileage_km",
    "km_per_year",
    "rating",
    "ext_body_pct",
    "engine_trans_pct",
    "suspension_pct",
    "interior_pct",
    "ac_heater_pct",
    "inspection_score",
    "engine_cc",
    "is_premium_brand",
    "is_automatic",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    listing_year = out["last_updated"].dt.year
    listing_year = listing_year.fillna(listing_year.max())
    out["car_age"] = (listing_year - out["model_year"]).clip(lower=0).astype(int)
    out["km_per_year"] = out["mileage_km"] / out["car_age"].clip(lower=1)
    out["inspection_score"] = out[cfg.INSPECTION_COLUMNS].mean(axis=1)
    out["is_premium_brand"] = out["brand"].isin(cfg.PREMIUM_BRANDS).astype(int)
    out["is_automatic"] = out["transmission"].eq("Automatic").astype(int)
    return out


def group_rare(values: pd.Series, min_count: int) -> pd.Series:
    """Replace categories with fewer than min_count rows by 'Other'."""
    counts = values.value_counts()
    rare = counts.index[counts < min_count]
    return values.where(~values.isin(rare), "Other")


def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    """One-hot encode. The most common level is left out and acts as the reference category."""
    counts = series.value_counts()
    reference = counts.index[0]
    levels = sorted(level for level in counts.index if level != reference)
    return pd.DataFrame(
        {f"{prefix}_{level}": (series == level).astype(int) for level in levels},
        index=series.index,
    )


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Final ML-ready table: identifiers, target, numeric features, and one-hot dummies.

    The most common category of each column is left out, so linear models do not hit a
    perfect-collinearity trap and the coefficients read as "compared with the typical car".
    """
    dummies = pd.concat([
        one_hot(df["fuel_type"], "fuel_type"),
        one_hot(group_rare(df["brand"], cfg.MIN_CATEGORY_COUNT), "brand"),
        one_hot(group_rare(df["city"], cfg.MIN_CATEGORY_COUNT), "city"),
    ], axis=1)
    return pd.concat([df[["listing_id", "car_name", cfg.TARGET] + MODEL_NUMERIC], dummies], axis=1)


def spearman_with_target(df: pd.DataFrame, columns: list, mask: pd.Series | None = None) -> dict:
    """Spearman rank correlation with price. Pass a boolean mask to restrict the rows."""
    subset = df if mask is None else df.loc[mask]
    return {column: float(subset[column].corr(subset[cfg.TARGET], method="spearman")) for column in columns}
