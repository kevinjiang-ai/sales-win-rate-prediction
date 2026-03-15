"""
Data pipeline: ingest CSV → load into SQLite → return clean DataFrame.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "crm.db"
CSV_PATH = DATA_DIR / "crm_data.csv"


def ingest_to_sqlite(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> None:
    """Load raw CSV into SQLite, replacing the table each run."""
    df = pd.read_csv(csv_path)
    con = sqlite3.connect(db_path)
    df.to_sql("deals", con, if_exists="replace", index=False)
    con.close()
    print(f"Ingested {len(df):,} rows into {db_path}")


def query_clean_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Pull data from SQLite and apply basic cleaning transformations:
    - Drop duplicates on deal_id
    - Clip outliers in deal_size and sales_cycle_days
    - Ensure binary flags are 0/1 integers
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM deals", con)
    con.close()

    # Deduplicate
    df = df.drop_duplicates(subset="deal_id")

    # Clip extreme outliers
    p99_deal = df["deal_size_usd"].quantile(0.99)
    df["deal_size_usd"] = df["deal_size_usd"].clip(upper=p99_deal)

    p99_cycle = df["sales_cycle_days"].quantile(0.99)
    df["sales_cycle_days"] = df["sales_cycle_days"].clip(upper=p99_cycle)

    # Ensure binary columns are int
    for col in ["demo_given", "proposal_sent", "existing_customer", "won"]:
        df[col] = df[col].astype(int)

    return df


def run_pipeline() -> pd.DataFrame:
    ingest_to_sqlite()
    df = query_clean_data()
    print(f"Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    run_pipeline()
