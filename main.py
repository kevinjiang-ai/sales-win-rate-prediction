"""
main.py — Run the full sales win-rate prediction pipeline end to end.

Steps:
  1. Generate 50K simulated CRM records
  2. Ingest CSV → SQLite → clean DataFrame
  3. Feature engineering + train/test split
  4. Train Logistic Regression and Gradient Boosting models
  5. Evaluate both models and save metrics + plots
"""

import json
from pathlib import Path

from data.generate_data import generate_crm_data
from src.data_pipeline import run_pipeline
from src.feature_engineering import encode_and_split
from src.model_training import train_gradient_boosting, train_logistic_regression
from src.evaluation import evaluate_model, save_metrics

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "crm_data.csv"


def main():
    print("=" * 60)
    print("  Sales Win-Rate Prediction Pipeline")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/5] Generating CRM data...")
    df_raw = generate_crm_data()
    df_raw.to_csv(CSV_PATH, index=False)
    print(f"  Saved {len(df_raw):,} records to {CSV_PATH}")

    # Step 2: Ingest to SQLite + clean
    print("\n[2/5] Running data pipeline (CSV → SQLite → clean DataFrame)...")
    df = run_pipeline()

    # Step 3: Feature engineering
    print("\n[3/5] Engineering features and splitting train/test...")
    X_train, X_test, y_train, y_test, feature_names, scaler = encode_and_split(df)

    # Step 4: Train models
    print("\n[4/5] Training models...")
    lr_model = train_logistic_regression(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    # Step 5: Evaluate
    print("\n[5/5] Evaluating models...")
    lr_metrics = evaluate_model(lr_model, X_test, y_test,
                                 "Logistic Regression", feature_names)
    gb_metrics = evaluate_model(gb_model, X_test, y_test,
                                 "Gradient Boosting", feature_names)

    all_metrics = [lr_metrics, gb_metrics]
    save_metrics(all_metrics)

    # Print summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'AUC-ROC':>8} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
    for m in all_metrics:
        print(f"  {m['model']:<25} {m['roc_auc']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>6.4f}")
    print("=" * 60)
    print("\nPipeline complete. Plots saved to reports/figures/")

    return all_metrics


if __name__ == "__main__":
    main()
