# Sales Win-Rate Prediction Pipeline

An end-to-end machine learning system that predicts the probability of closing a B2B sales deal. Built with 50,000 simulated CRM records, a full ETL pipeline, and two production-grade models.

---

## Overview

Sales teams generate vast amounts of interaction data — meetings, emails, calls, competitor intel — but rarely have a systematic way to estimate which deals will close. This project builds a complete ML pipeline that ingests that data, engineers predictive features, trains baseline and advanced models, and produces calibrated win-probability scores.

**Use case:** Score open pipeline deals in real time to prioritize rep effort and forecast revenue.

---

## Pipeline Architecture

```
data/crm_data.csv
      │
      ▼
[1] data_pipeline.py     ← Ingest CSV → SQLite → clean DataFrame
      │
      ▼
[2] feature_engineering.py  ← Encode categoricals, derive features, train/test split
      │
      ▼
[3] model_training.py    ← Logistic Regression + Gradient Boosting
      │
      ▼
[4] evaluation.py        ← AUC-ROC, precision, recall, calibration, confusion matrix
```

---

## Results

| Model                 | AUC-ROC | Precision | Recall | F1    |
|-----------------------|---------|-----------|--------|-------|
| Logistic Regression   | 0.7061  | 0.9322    | 1.0000 | 0.9649|
| Gradient Boosting     | 0.6893  | 0.9323    | 0.9997 | 0.9648|

> **Logistic Regression wins on AUC-ROC (0.706)**, making it the preferred model for probability calibration and ranking deals by win likelihood. Gradient Boosting provides feature importance rankings.

All evaluation plots (ROC curves, PR curves, calibration curves, confusion matrices, feature importances) are saved to `reports/figures/`.

---

## Dataset

50,000 simulated CRM records with 21 features:

| Feature Category | Features |
|-----------------|----------|
| Deal metadata   | deal_size_usd, contract_length_months, industry, region |
| Sales activity  | num_meetings, num_emails, num_calls, sales_cycle_days |
| Process flags   | demo_given, proposal_sent |
| Competition     | competitor_count, discount_pct |
| Rep profile     | rep_tenure_years, rep_quota_attainment, rep_win_rate_historical |
| Customer        | company_size, existing_customer, crm_score, lead_source |

**Win rate:** ~93% (class imbalance reflects typical enterprise CRM data where deals with full activity logs skew toward closed-won).

---

## Feature Engineering

- `log_deal_size` — log-transform of skewed deal size
- `activity_density` — total touches / sales cycle days
- `engagement_score` — weighted sum of meetings, calls, emails
- `high_competition` — binary flag for 3+ competitors
- One-hot encoding of all categorical columns

---

## Project Structure

```
sales-win-rate-prediction/
├── data/
│   └── generate_data.py       # Simulate 50K CRM records
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA with visualizations
├── src/
│   ├── data_pipeline.py       # CSV → SQLite → clean DataFrame
│   ├── feature_engineering.py # Feature creation + train/test split
│   ├── model_training.py      # LR + GBM training, model serialization
│   └── evaluation.py          # Metrics + plot generation
├── main.py                    # End-to-end runner
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11+ |
| Data storage | SQLite (via sqlite3) |
| Data manipulation | Pandas, NumPy |
| ML models | Scikit-learn (LogisticRegression, GradientBoostingClassifier) |
| Evaluation | sklearn.metrics (AUC-ROC, PR curve, calibration) |
| Visualization | Matplotlib |
| Notebook | Jupyter |

---

## Author

**Kevin Jiang** — Machine Learning & Data Analytics Engineer
Deep Simplicity LLC | Purdue University '26
[github.com/kevinjiang-ai](https://github.com/kevinjiang-ai)
