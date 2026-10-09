"""Settings for the PakWheels cleaning pipeline.

Change behaviour here instead of inside the code.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pakwheeldataset.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pakwheels_clean.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_report.md"

TARGET = "price_pkr"

# ---- Parsing ---------------------------------------------------------------
# Each ad is scraped on several dates. True keeps only the most recently updated copy of each ad.
REMOVE_REPEAT_SCRAPES = True

LAC_PKR = 100_000          # 1 lac   = 100,000 PKR
CRORE_PKR = 10_000_000     # 1 crore = 10,000,000 PKR

# The source only ever records "Automatic". Blank gearbox cells belong to manual cars
# (see README, Step 1).
BLANK_TRANSMISSION_VALUE = "Manual"

# Electric models have a blank fuel type in the source. These car-name patterns identify them.
ELECTRIC_NAME_PATTERN = r"\b(?:EQ[CES]|e-tron|i7|Leaf|Tesla|EV|Electric|Model [3SXY]|MG 4)\b"

INSPECTION_COLUMNS = [
    "ext_body_pct",
    "engine_trans_pct",
    "suspension_pct",
    "interior_pct",
    "ac_heater_pct",
]

# ---- Missing values ----------------------------------------------------------
# Method per column: "mode" (categorical), "median", "mean", or "knn" (engine_cc only).
MISSING_VALUE_METHODS = {
    "fuel_type": "mode",
    "ac_heater_pct": "median",   # most cars score exactly 100%, so the median is the robust choice
    "engine_cc": "knn",          # engine size depends on model, age, mileage and fuel
}
KNN_NEIGHBORS = 3                # chosen by comparing hold-out accuracy for k = 3 and k = 5 during development

# ---- Outliers ----------------------------------------------------------------
IQR_MULTIPLIER = 1.5
# "log" = IQR fences computed on a log scale (for right-skewed columns); "raw" = plain IQR.
OUTLIER_COLUMNS = {
    "price_pkr": "log",
    "mileage_km": "log",
    "km_per_year": "log",
    "rating": "raw",
    "ext_body_pct": "raw",
    "engine_trans_pct": "raw",
    "interior_pct": "raw",
    "suspension_pct": "raw",
    "ac_heater_pct": "raw",
    "inspection_score": "raw",
}
# Checked and reported but never capped: column -> (reason, pandas query limiting the rows or None)
OUTLIER_EXCLUDED = {
    "car_age": (
        "age in whole years; the flagged cars are genuinely old, and capping would make them look younger",
        None,
    ),
    "engine_cc": (
        "engine sizes are discrete classes (1000, 1500, 2500 cc...); IQR fences would cut real classes",
        "fuel_type != 'Electric'",
    ),
}
TREAT_TARGET_OUTLIERS = True     # False = keep real luxury prices untouched

# ---- Feature engineering -----------------------------------------------------
PREMIUM_BRANDS = [
    "Audi", "BMW", "Jaguar", "Lexus", "Mercedes", "Porsche", "Range", "Tesla", "Volvo",
]                                # "Range" = Range Rover
MIN_CATEGORY_COUNT = 10          # rarer brands and cities are grouped as "Other" before one-hot encoding
