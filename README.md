# RetailCast — Store Sales Forecasting

> **An end-to-end Machine Learning system that predicts daily retail store sales using XGBoost, deployed via a Flask web application.**

---

## What This Project Does

Given information about a store and a specific day, the model predicts **how much that store will sell (in €)**. This helps businesses with:
- 📦 Ordering the right amount of stock
- 👥 Scheduling the right number of staff
- 🏷️ Planning promotions on the right days
- 📊 Setting realistic revenue targets

---

## Dataset

**Source:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales)

| File | Description | Size |
|---|---|---|
| `train.csv` | Daily sales diary — one row per store per day | ~1M rows |
| `store.csv` | Store profiles — one row per store | 1,115 rows |

Both files are merged on `Store ID` to create a single dataset of **844,338 rows** after filtering to open stores with sales > 0.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, NumPy |
| Machine learning | XGBoost, scikit-learn |
| Model saving | joblib |
| Visualisation | Matplotlib, Seaborn |
| Web backend | Flask |
| Web frontend | HTML, CSS, JavaScript |
| Development | Jupyter Notebook |

---

## Features Used by the Model (21 total)

**Date features** — Year, Month, Day, WeekOfYear, Quarter, IsWeekend, IsMonthStart, IsMonthEnd

**Store features** — Store ID, StoreType, Assortment, CompetitionDistance

**Promotion & holiday features** — Promo, Promo2, StateHoliday, SchoolHoliday

**Interaction features (engineered)** — Promo × Weekend, Promo × Month, Store × DayOfWeek, NearCompetitor (<1km)

---

## Model — XGBoost Regressor

XGBoost builds 1,000 decision trees where each tree learns from the mistakes of the previous one. Key settings:

```python
XGBRegressor(
    n_estimators    = 1000,   # number of trees
    learning_rate   = 0.03,   # careful, small steps
    max_depth       = 8,      # depth of each tree
    subsample       = 0.85,
    colsample_bytree= 0.75,
    min_child_weight= 5,      # prevents overfitting
    reg_alpha       = 0.1,    # L1 regularisation
    reg_lambda      = 1.0     # L2 regularisation
)
```

---

## Model Results (KPIs)

| KPI | Value | Meaning |
|---|---|---|
| **R² Score** | 0.8141 | Model explains 81% of sales variation |
| **RMSE** | €1,339 | Average prediction error |
| **MAE** | €964 | Typical daily error (~14% of avg sales) |
| **MAPE** | 15.7% | Mean % error per prediction |
| **Within 10%** | 45.3% | Predictions within 10% of actual |
| **Within 20%** | ~71% | Predictions within 20% of actual |

> Average actual daily sales across the dataset: **€6,956**

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install flask xgboost scikit-learn pandas numpy matplotlib seaborn joblib
```

### Step 2 — Download the dataset
Download `train.csv` and `store.csv` from [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data) and place them in the project folder.

### Step 3 — Train the model
Open `sales_forecast.ipynb` in Jupyter Notebook or Google Colab and run all cells top to bottom. This will generate `sales_model.pkl`.

### Step 4 — Start the Flask server
```bash
cd RetailCast
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
* Debug mode: on
```

### Step 5 — Open the web app
Go to your browser and visit:
```
http://127.0.0.1:5000
```

> ⚠️ Do **not** open `index.html` directly by double-clicking. Always access it through Flask at `http://127.0.0.1:5000`.

---

## Using the Web Interface

The form is split into three sections:

| Section | Fields |
|---|---|
| 🏪 Store Information | Store ID, Store Type, Assortment Level, Competition Distance |
| 📅 Date & Time | Year, Month, Day, Week of Year, Day of Week, Quarter, Weekend, Month Start/End |
| 🏷️ Promotions & Holidays | Promo, Promo2, State Holiday, School Holiday |

Fill in all fields and click **Predict Sales**. The estimated daily revenue will appear instantly below the form.

Five features (`Promo_Weekend`, `Promo_Month`, `Store_DayOfWeek`, `NearCompetitor`, `Quarter`) are **computed automatically** inside `app.py` — you do not need to enter them manually.

---

## How KPIs Support Business Decisions

| KPI | Business Use |
|---|---|
| **R² (81%)** | Confirms the model reliably explains sales patterns |
| **MAE (€964)** | Sets expectations for forecast accuracy in planning |
| **MAPE (15.7%)** | Used to calculate safety stock buffers |
| **Within 10%** | Indicates on which days predictions are most reliable |

---

## Future Improvements

- Add **time-series cross-validation** (`TimeSeriesSplit`) for more realistic evaluation
- Try **LightGBM** or **Prophet** and compare results
- Add **confidence intervals** to predictions (upper/lower bounds)
- Add **store-level lag features** (yesterday's sales, last week's sales)
- Deploy to **cloud** (Railway, Render, or HuggingFace Spaces — all free tiers available)
- Add **automated retraining** pipeline when new sales data arrives

---

## Source & Credits

- **Dataset:** Rossmann Store Sales, Kaggle
- **Model:** XGBoost (Chen & Guestrin, 2016)
- **References:** U.S. Bureau of Labor Statistics, LinkedIn Job Trends 2025, Gartner Cloud Report 2025

---

*RetailCast · Built with Python, XGBoost & Flask · April 2026*
