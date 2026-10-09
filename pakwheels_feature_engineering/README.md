# PakWheels: Advanced EDA & Feature Engineering

Turns the raw PakWheels used-car listings (`pakwheeldataset.csv`) into a clean, ML-ready table.
It covers statistical imputation (mean, median, KNN), IQR outlier capping, and feature engineering.

## Project layout

```
pakwheels_feature_engineering/
├── data/
│   ├── raw/pakwheeldataset.csv         your original file (not modified)
│   └── processed/pakwheels_clean.csv   OUTPUT: cleaned, ML-ready table
├── reports/
│   └── data_quality_report.md          OUTPUT: evidence for every decision
├── src/
│   ├── config.py          all settings (paths, methods, thresholds)
│   ├── parsing.py         step 1: text to numbers, remove repeat scrapes, recover facts
│   ├── imputation.py      step 2: mode / median / mean / KNN imputation
│   ├── outliers.py        step 3: IQR detection and capping
│   ├── features.py        step 4: engineered features and one-hot encoding
│   └── run_pipeline.py    runs all four steps and writes both outputs
├── requirements.txt
└── README.md
```

## Run it in VS Code

1. **File → Open Folder** and choose `pakwheels_feature_engineering`.
2. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS / Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install the libraries: `pip install -r requirements.txt`
4. Pick the environment: **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**.
5. Run the pipeline: `python -m src.run_pipeline`
   (or open `src/run_pipeline.py` and click **Run**).
6. Open `reports/data_quality_report.md` (Ctrl+Shift+V for preview) and `data/processed/pakwheels_clean.csv`.
7. For charts, open `notebooks/eda_charts.ipynb`, choose the `.venv` kernel if prompted, then select **Run All**. The notebook also includes saved chart outputs, so they should appear when you open it. If VS Code asks, install its Microsoft **Jupyter** extension.

Requires Python 3.9 or newer.

## What each step does

### Step 1: Parsing (`parsing.py`)
- Converts text to numbers: `PKR 49.5 lacs` and `PKR 1 crore` to PKR, `16,603 km` to km, `9.7/10` to a rating, `89%` to a score.
- Removes repeat scrapes (`REMOVE_REPEAT_SCRAPES` in `config.py`). The same ad appears several times with the same ad ID, sometimes with a new price. The most recently updated copy is kept, so 1,874 rows become 1,001 ads.
- Recovers facts that the row already states:
  - **Model year** is blank for some ads, but every car name ends with the year. It matches `Model Date` in every row where both exist.
  - **Electric cars** have a blank fuel type. Names such as Tesla, e-tron, Leaf, and EQC identify them.
  - **Blank gearbox** becomes `Manual`. The source only ever records `Automatic`. Blank rows are never hybrid or electric, and none mention an automatic gearbox. Filling them statistically would have turned manual cars into automatic ones.

### Step 2: Missing values (`imputation.py`)

| Column | Missing (after step 1) | Method | Why |
|---|---|---|---|
| `fuel_type` | 5 | mode | categorical |
| `ac_heater_pct` | 4 | median | most cars score exactly 100%, so the median is the robust choice |
| `engine_cc` | 525 | KNN (k = 3) | engine size depends on model, age, mileage and fuel |

- **KNN** finds the 3 most similar petrol, diesel or hybrid cars (same model family, model year, mileage and fuel) and averages their engine sizes. Electric cars are never used as donors.
- The report checks KNN by hiding known engine sizes and predicting them again. KNN lands within 100 cc about 79% of the time, against 32% for a single median value.
- **Mean** is also available. Set a column to `"mean"` in `src/config.py` to use it.

### Step 3: Outliers (`outliers.py`)
- **IQR rule:** a value below `Q1 - 1.5 x IQR` or above `Q3 + 1.5 x IQR` is an outlier. It is capped to that fence, so the row is kept but the extreme value stops dominating the model.
- Right-skewed columns (price, mileage, km per year) use fences on a log scale.
- Columns where the IQR is 0 (for example, suspension at 100%) are skipped. The rule would otherwise flag every value below 100.
- Car age and engine size are checked but not capped. Their "outliers" are real values, such as 1000 cc and 5000 cc engines.
- **Price (the target)** is capped by default. Set `TREAT_TARGET_OUTLIERS = False` in `config.py` to keep real luxury prices untouched.

### Step 4: Feature engineering (`features.py`)

| Feature | How it is made |
|---|---|
| `car_age` | listing year minus model year (never below 0) |
| `km_per_year` | mileage divided by car age |
| `inspection_score` | average of the five inspection percentages |
| `engine_cc` | engine size read from the car name (`1.5` to 1500); electric cars 0; gaps filled by KNN |
| `is_premium_brand` | 1 for premium brands such as BMW, Audi and Mercedes |
| `is_automatic` | 1 if the gearbox is Automatic |
| `city` | taken from the listing URL, then one-hot encoded |

Categorical columns (`fuel_type`, `brand`, `city`) are one-hot encoded. Rare brands and cities are grouped as `Other`. The most common category in each group is left out as the reference.

## Output table

- `listing_id` and `car_name` are identifiers. **Drop them before training.**
- `price_pkr` is the target.
- Everything else is a feature.

## Adjusting the pipeline

Edit `src/config.py`. Examples:
- `REMOVE_REPEAT_SCRAPES = False` to keep all 1,874 rows.
- `MISSING_VALUE_METHODS["engine_cc"] = "median"` to compare methods.
- `IQR_MULTIPLIER = 3.0` to flag only extreme outliers.
- `TREAT_TARGET_OUTLIERS = False` to keep the real prices.

## Before you train a model

Imputation values and outlier fences are calculated on the full dataset here, which is fine for exploration. For modelling, calculate them on the training split only and apply them to the test split. This avoids leaking test information into training.


## Author

**Laiba Aslam**  
BS Information Technology (BSIT) Student  
University of Sindh, Jamshoro  

**Interests:** Data Science, Data Analytics, and Machine Learning.

**Project:** PakWheels — Advanced Exploratory Data Analysis (EDA) & Feature Engineering.