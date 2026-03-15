"""
Generate 50,000 simulated CRM records for sales win-rate prediction.
Saves to data/crm_data.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 50_000

def generate_crm_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Deal metadata ---
    deal_ids = [f"DEAL-{i:06d}" for i in range(n)]
    industries = rng.choice(
        ["SaaS", "Manufacturing", "Healthcare", "Finance", "Retail", "Logistics"],
        size=n, p=[0.25, 0.20, 0.18, 0.17, 0.12, 0.08]
    )
    regions = rng.choice(["North America", "Europe", "APAC", "LATAM"], size=n,
                         p=[0.45, 0.28, 0.18, 0.09])
    deal_size = np.round(rng.lognormal(mean=10.5, sigma=1.2, size=n), 2)
    contract_length = rng.choice([12, 24, 36, 48, 60], size=n,
                                  p=[0.30, 0.35, 0.20, 0.10, 0.05])

    # --- Sales process features ---
    num_stakeholders = rng.integers(1, 12, size=n)
    sales_cycle_days = rng.integers(7, 365, size=n)
    num_meetings = rng.integers(0, 20, size=n)
    num_emails = rng.integers(0, 80, size=n)
    num_calls = rng.integers(0, 30, size=n)
    competitor_count = rng.integers(0, 6, size=n)
    demo_given = rng.choice([0, 1], size=n, p=[0.3, 0.7])
    proposal_sent = rng.choice([0, 1], size=n, p=[0.25, 0.75])
    discount_pct = np.clip(rng.normal(10, 8, size=n), 0, 40).round(1)

    # --- Rep features ---
    rep_tenure_years = np.clip(rng.normal(4, 2.5, size=n), 0.5, 15).round(1)
    rep_quota_attainment = np.clip(rng.normal(0.95, 0.20, size=n), 0.2, 1.8).round(2)
    rep_win_rate_historical = np.clip(rng.normal(0.42, 0.12, size=n), 0.05, 0.90).round(3)

    # --- Lead source ---
    lead_source = rng.choice(
        ["Inbound", "Outbound", "Referral", "Partner", "Event"],
        size=n, p=[0.30, 0.25, 0.22, 0.15, 0.08]
    )

    # --- Customer features ---
    company_size = rng.choice(
        ["SMB", "Mid-Market", "Enterprise", "Large Enterprise"],
        size=n, p=[0.25, 0.35, 0.28, 0.12]
    )
    existing_customer = rng.choice([0, 1], size=n, p=[0.70, 0.30])
    crm_score = np.clip(rng.normal(65, 20, size=n), 0, 100).round(1)

    # --- Construct win probability using a realistic logit model ---
    log_deal_size = np.log1p(deal_size)

    industry_effect = np.where(np.isin(industries, ["SaaS", "Finance"]), 0.3,
                     np.where(np.isin(industries, ["Healthcare"]), 0.1, -0.1))
    lead_effect = np.where(lead_source == "Referral", 0.5,
                  np.where(lead_source == "Inbound", 0.2,
                  np.where(lead_source == "Partner", 0.15, 0.0)))
    size_effect = np.where(company_size == "Enterprise", 0.25,
                  np.where(company_size == "Large Enterprise", 0.15,
                  np.where(company_size == "Mid-Market", 0.05, -0.1)))

    logit = (
        -1.2
        + 0.15 * log_deal_size
        - 0.003 * sales_cycle_days
        + 0.08 * num_meetings
        + 0.005 * num_emails
        + 0.015 * num_calls
        - 0.12 * competitor_count
        + 0.35 * demo_given
        + 0.40 * proposal_sent
        - 0.015 * discount_pct
        + 0.10 * rep_tenure_years
        + 0.80 * rep_win_rate_historical
        + 0.60 * rep_quota_attainment
        + 0.005 * crm_score
        + 0.50 * existing_customer
        - 0.04 * num_stakeholders
        + industry_effect
        + lead_effect
        + size_effect
        + rng.normal(0, 0.5, size=n)   # noise
    )

    prob_win = 1 / (1 + np.exp(-logit))
    won = rng.binomial(1, prob_win).astype(int)

    df = pd.DataFrame({
        "deal_id": deal_ids,
        "industry": industries,
        "region": regions,
        "deal_size_usd": deal_size,
        "contract_length_months": contract_length,
        "num_stakeholders": num_stakeholders,
        "sales_cycle_days": sales_cycle_days,
        "num_meetings": num_meetings,
        "num_emails": num_emails,
        "num_calls": num_calls,
        "competitor_count": competitor_count,
        "demo_given": demo_given,
        "proposal_sent": proposal_sent,
        "discount_pct": discount_pct,
        "rep_tenure_years": rep_tenure_years,
        "rep_quota_attainment": rep_quota_attainment,
        "rep_win_rate_historical": rep_win_rate_historical,
        "lead_source": lead_source,
        "company_size": company_size,
        "existing_customer": existing_customer,
        "crm_score": crm_score,
        "won": won,
    })

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "crm_data.csv"
    df = generate_crm_data()
    df.to_csv(output_path, index=False)
    win_rate = df["won"].mean()
    print(f"Generated {len(df):,} records → {output_path}")
    print(f"Win rate: {win_rate:.1%}  |  Wins: {df['won'].sum():,}  |  Losses: {(~df['won'].astype(bool)).sum():,}")
