"""IBM Telco customer churn — download, clean, engineer, and split into branch groups.

The dataset is the public churn benchmark (7,043 customers, 21 columns, no auth).
Its columns partition naturally into the three feature groups the fusion network
expects: billing (contract and money), profile (who the customer is), and services
(what they actually use). A few engineered features are added where the raw table
is too flat to give the selection funnel anything to prune — each is derived only
from same-row values, so nothing leaks across the train/test split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_CACHE = Path("data/telco.csv")

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

GROUPS: dict[str, list[str]] = {
    # ordered by priority: stage-6 cross-group dedup keeps earlier groups' concepts
    "billing": [
        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
        "TotalCharges", "avg_monthly_spend", "charges_drift",
    ],
    "profile": [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "tenure_bucket",
    ],
    "services": SERVICE_COLS + ["active_services", "streaming_bundle", "protection_bundle"],
}

TARGET = "Churn"


def load_telco(cache_path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the Telco CSV, downloading once into the local cache."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(TELCO_URL)
        df.to_csv(cache_path, index=False)
    return pd.read_csv(cache_path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the dataset's known quirks and add derived features.

    TotalCharges is a string column with 11 blank values (customers with tenure 0);
    it becomes numeric with blanks as 0. Engineered features:

    - avg_monthly_spend:  TotalCharges / tenure — long-run spend rate
    - charges_drift:      MonthlyCharges - avg_monthly_spend — is the bill trending up?
    - tenure_bucket:      categorical tenure band (new/1yr/2yr/3yr/veteran)
    - active_services:    count of subscribed service lines
    - streaming_bundle:   has both StreamingTV and StreamingMovies
    - protection_bundle:  has both OnlineBackup and DeviceProtection
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    tenure = df["tenure"].replace(0, 1)
    df["avg_monthly_spend"] = (df["TotalCharges"] / tenure).round(2)
    df["charges_drift"] = (df["MonthlyCharges"] - df["avg_monthly_spend"]).round(2)
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)

    active = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["active_services"] = active
    df["streaming_bundle"] = np.where(
        (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes"), "Yes", "No")
    df["protection_bundle"] = np.where(
        (df["OnlineBackup"] == "Yes") & (df["DeviceProtection"] == "Yes"), "Yes", "No")

    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_dataset(cache_path: Path = DEFAULT_CACHE) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return the cleaned frame plus the group -> columns mapping."""
    df = clean_and_engineer(load_telco(cache_path))
    groups = {name: [c for c in cols if c in df.columns] for name, cols in GROUPS.items()}
    return df, groups
