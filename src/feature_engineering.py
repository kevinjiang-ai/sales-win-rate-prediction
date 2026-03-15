"""
Feature engineering: encode categoricals, create derived features,
and split into train/test sets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ["industry", "region", "lead_source", "company_size"]
TARGET = "won"
DROP_COLS = ["deal_id", "won"]

RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features before encoding."""
    df = df.copy()

    # Interaction: activity density = total touches / sales cycle
    df["activity_density"] = (
        (df["num_meetings"] + df["num_calls"] + df["num_emails"])
        / (df["sales_cycle_days"] + 1)
    )

    # Log-transform skewed monetary column
    df["log_deal_size"] = np.log1p(df["deal_size_usd"])

    # Engagement score
    df["engagement_score"] = (
        df["num_meetings"] * 3 + df["num_calls"] * 2 + df["num_emails"]
    )

    # Competitiveness flag
    df["high_competition"] = (df["competitor_count"] >= 3).astype(int)

    return df


def encode_and_split(
    df: pd.DataFrame, test_size: float = 0.20, scale: bool = True
):
    """
    One-hot encode categoricals, split train/test, optionally scale numerics.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler (or None)
    """
    df = engineer_features(df)

    y = df[TARGET].values
    X = df.drop(columns=DROP_COLS)

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=False)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(
        f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | "
        f"Features: {X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test, feature_names, scaler
