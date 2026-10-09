"""Step 2: fill the remaining missing values with statistical imputation.

Methods used: mode (categorical), median, mean, and KNN (engine size).
Each statistic is computed from the cars that have the value. Price is never used to fill a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

STATISTICS = {
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "mode": lambda s: s.mode(dropna=True).iloc[0],
}


def knn_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Columns used to find similar cars. Price is deliberately excluded (it would leak the target)."""
    numeric = pd.DataFrame({
        "model_year": df["model_year"].astype(float),
        "log_mileage": np.log1p(df["mileage_km"]),
    })
    numeric = (numeric - numeric.mean()) / numeric.std()   # same scale, so no column dominates
    categorical = pd.get_dummies(df[["model_key", "fuel_type"]], dtype=float)
    return pd.concat([numeric, categorical], axis=1)


def fill_engine_knn(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Fill missing engine sizes with the average of the k most similar combustion cars."""
    out = df.copy()
    combustion = out["fuel_type"].ne("Electric")   # electric cars have no engine, so they are not donors
    rows = out.loc[combustion]
    predictors = knn_predictors(rows)
    matrix = np.column_stack([rows["engine_cc"].to_numpy(float), predictors.to_numpy(float)])
    filled = KNNImputer(n_neighbors=k).fit_transform(matrix)[:, 0]

    missing = rows["engine_cc"].isna().to_numpy()
    out.loc[rows.index[missing], "engine_cc"] = np.round(filled[missing], -2)   # nearest 100 cc
    return out


def validate_engine_knn(df: pd.DataFrame, k: int, share: float = 0.2, splits: int = 5) -> dict:
    """Hide 20% of the known engine sizes, predict them, and compare with a plain median guess.

    Repeated over several random splits and averaged, so one lucky or unlucky draw does not decide the result.
    """
    combustion = df["fuel_type"].ne("Electric")
    known = df.index[combustion & df["engine_cc"].notna()].to_numpy()
    runs = []
    for seed in range(splits):
        rng = np.random.default_rng(seed)
        hidden = rng.choice(known, size=int(round(share * len(known))), replace=False)

        trial = df.copy()
        trial.loc[hidden, "engine_cc"] = np.nan
        predicted = fill_engine_knn(trial, k).loc[hidden, "engine_cc"].to_numpy(float)
        actual = df.loc[hidden, "engine_cc"].to_numpy(float)
        median_guess = trial.loc[combustion, "engine_cc"].median()

        knn_error = np.abs(predicted - actual)
        median_error = np.abs(median_guess - actual)
        runs.append([
            len(hidden),
            knn_error.mean(), (knn_error <= 100).mean(),
            median_error.mean(), (median_error <= 100).mean(),
        ])

    hidden_avg, knn_mae, knn_within, median_mae, median_within = np.mean(runs, axis=0)
    return {
        "hidden": int(round(hidden_avg)),
        "splits": splits,
        "knn_mae": float(knn_mae),
        "knn_within_100": float(knn_within),
        "median_mae": float(median_mae),
        "median_within_100": float(median_within),
    }


def impute_missing(df: pd.DataFrame, plan: dict, k: int) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """Apply the imputation plan. Returns the filled table, a summary table, and the KNN check."""
    out = df.copy()
    rows = []
    knn_check = None

    # Mode first, because the engine KNN uses fuel_type as a predictor.
    for column, method in sorted(plan.items(), key=lambda item: item[1] != "mode"):
        row = {
            "Column": column,
            "Method": method,
            "Missing filled": int(out[column].isna().sum()),
            "Mean of known": np.nan,
            "Median of known": np.nan,
            "Skew of known": np.nan,
            "Filled with": "",
        }

        if method == "knn":
            if column != "engine_cc":
                raise ValueError("KNN imputation is set up for engine_cc only.")
            knn_check = validate_engine_knn(out, k)
            out = fill_engine_knn(out, k)
            row["Method"] = f"KNN (k={k})"
            row["Filled with"] = "average of the most similar cars, rounded to 100 cc"
        elif method in STATISTICS:
            known = out[column].dropna()
            if pd.api.types.is_numeric_dtype(out[column]):
                row["Mean of known"] = float(known.mean())
                row["Median of known"] = float(known.median())
                row["Skew of known"] = float(known.skew())
            value = STATISTICS[method](out[column])
            out[column] = out[column].fillna(value)
            row["Filled with"] = value if method == "mode" else round(float(value), 2)
        else:
            raise ValueError(f"Unknown imputation method '{method}' for {column}.")
        rows.append(row)

    return out, pd.DataFrame(rows), knn_check
