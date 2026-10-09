"""Step 1: turn raw PakWheels text into numbers and clean categories.

Handles price text ("PKR 49.5 lacs", "PKR 1 crore"), mileage text ("16,603 km"),
rating text ("9.7/10"), score text ("89%"), blank and "N/A" cells, model years that
are blank but printed in the car name, and ads that were scraped more than once.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config as cfg

RAW_TO_CLEAN = {
    "Car Name": "car_name",
    "Rating": "rating",
    "Exterior & Body": "ext_body_pct",
    "Engine/Transmission/Clutch": "engine_trans_pct",
    "Suspension/Steering": "suspension_pct",
    "Interior": "interior_pct",
    "AC/Heater": "ac_heater_pct",
    "Price": "price_pkr",
    "Model Date": "model_year",
    "Mileage": "mileage_km",
    "Fuel Type": "fuel_type",
    "Transmission": "transmission",
    "Last Updated Date": "last_updated",
    "Offer URL": "offer_url",
}

_PRICE_RE = re.compile(r"PKR\s*([\d,]*\.?\d+)\s*(lacs?|crore)", re.IGNORECASE)
_KM_RE = re.compile(r"([\d,]+)\s*km", re.IGNORECASE)
_RATING_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_AD_ID_RE = re.compile(r"-(\d+)$")
_CITY_RE = re.compile(r"-for-sale-in-([a-z-]+)-\d+$")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})$")
_LITRES_RE = re.compile(r"(?<![\d.])(\d\.\d)(?!\d)")   # the "1.5" in "Honda City 1.5L CVT"
_ENGINE_TOKEN_RE = re.compile(r"\d\.\d\w*")             # a whole word such as "1.5L" or "2.0T"
_AUTO_HINT_RE = re.compile(r"CVT|A/T|AGS|DCT|Automatic", re.IGNORECASE)


def load_raw(path) -> pd.DataFrame:
    """Read every cell as text so that no value is changed silently."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    missing = [name for name in RAW_TO_CLEAN if name not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    raw = raw[list(RAW_TO_CLEAN)].apply(lambda col: col.str.strip())
    return raw.rename(columns=RAW_TO_CLEAN)


def _number(text: str, pattern: re.Pattern) -> float:
    match = pattern.search(text)
    return float(match.group(1).replace(",", "")) if match else np.nan


def _price_pkr(text: str) -> float:
    match = _PRICE_RE.search(text)
    if not match:
        return np.nan
    unit = cfg.LAC_PKR if match.group(2).lower().startswith("lac") else cfg.CRORE_PKR
    return float(match.group(1).replace(",", "")) * unit


def _engine_cc(name: str) -> float:
    match = _LITRES_RE.search(name)
    return round(float(match.group(1)) * 1000) if match else np.nan


def _model_key(name: str) -> str:
    """'Toyota Corolla GLi 1.3 VVTi 2019' -> 'Toyota Corolla GLi'. Used to find similar cars."""
    words = _YEAR_RE.sub("", name).split()
    words = [word for word in words if not _ENGINE_TOKEN_RE.fullmatch(word)]
    return " ".join(words[:3])


def _blank_to_nan(column: pd.Series) -> pd.Series:
    return column.replace({"": np.nan, "N/A": np.nan})


def parse_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert each raw column into a number, a category, or a date."""
    name = raw["car_name"].str.replace(r"\s+", " ", regex=True)
    url = raw["offer_url"]

    ad_id = url.str.extract(_AD_ID_RE)[0]
    if ad_id.isna().any():
        raise ValueError("Some Offer URLs do not end with an ad ID, so repeat scrapes cannot be matched.")

    df = pd.DataFrame({
        "listing_id": ad_id.astype(int),
        "car_name": name,
        "brand": name.str.split().str[0],
        "model_key": name.map(_model_key),
        "city": url.str.extract(_CITY_RE)[0].str.replace("-", " ").str.title(),
        "price_pkr": raw["price_pkr"].map(_price_pkr),
        "mileage_km": raw["mileage_km"].map(lambda text: _number(text, _KM_RE)),
        "rating": raw["rating"].map(lambda text: _number(text, _RATING_RE)),
        "model_year": pd.to_numeric(_blank_to_nan(raw["model_year"]), errors="coerce"),
        "name_year": name.str.extract(_YEAR_RE)[0].astype(float),
        "fuel_type": _blank_to_nan(raw["fuel_type"]),
        "transmission": _blank_to_nan(raw["transmission"]),
        "last_updated": pd.to_datetime(
            _blank_to_nan(raw["last_updated"]), format="%b %d, %Y", errors="coerce"
        ),
        "engine_cc": name.map(_engine_cc),
    })
    for column in cfg.INSPECTION_COLUMNS:
        df[column] = raw[column].map(lambda text: _number(text, _PERCENT_RE))
    return df


def drop_repeated_listings(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Each ad is scraped on several dates. Keep the most recently updated copy of each ad."""
    ordered = df.sort_values(["listing_id", "last_updated"], na_position="first")
    unique = ordered.drop_duplicates("listing_id", keep="last").reset_index(drop=True)
    return unique, len(df) - len(unique)


def recover_from_text(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill gaps that other columns or the car name already answer exactly.

    Returns the updated table and a dict of counts used in the report.
    """
    df = df.copy()
    facts: dict = {}

    both_years = df["model_year"].notna() & df["name_year"].notna()
    facts["year_agreement"] = float((df.loc[both_years, "model_year"] == df.loc[both_years, "name_year"]).mean())
    year_gap = df["model_year"].isna()
    df["model_year"] = df["model_year"].fillna(df["name_year"])
    facts["year_recovered"] = int((year_gap & df["model_year"].notna()).sum())

    is_electric_model = df["car_name"].str.contains(cfg.ELECTRIC_NAME_PATTERN, regex=True)
    fuel_gap = is_electric_model & df["fuel_type"].isna()
    df.loc[fuel_gap, "fuel_type"] = "Electric"
    facts["ev_fuel_recovered"] = int(fuel_gap.sum())

    electric = df["fuel_type"].eq("Electric")
    df.loc[electric, "engine_cc"] = 0          # electric cars have no engine
    facts["ev_engine_zero"] = int(electric.sum())

    gearbox_gap = df["transmission"].isna()
    facts["gearbox_blank"] = int(gearbox_gap.sum())
    facts["gearbox_blank_fuel"] = df.loc[gearbox_gap, "fuel_type"].fillna("blank").value_counts().to_dict()
    facts["gearbox_blank_auto_hint"] = int(df.loc[gearbox_gap, "car_name"].str.contains(_AUTO_HINT_RE).sum())
    facts["gearbox_blank_manual_hint"] = int(
        df.loc[gearbox_gap, "car_name"].str.contains(r"M/T|\bMT\b|Manual", case=False, regex=True).sum()
    )
    df["transmission"] = df["transmission"].fillna(cfg.BLANK_TRANSMISSION_VALUE)
    return df, facts
