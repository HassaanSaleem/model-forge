import pandas as pd

from model_forge.churn.data import GROUPS, SERVICE_COLS, TARGET, clean_and_engineer


def raw_frame() -> pd.DataFrame:
    base = {
        "customerID": ["a", "b", "c"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [0, 12, 48],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        "MonthlyCharges": [70.0, 55.5, 90.0],
        "TotalCharges": [" ", "666.0", "4320.0"],  # the dataset's blank-string quirk
        "Churn": ["Yes", "No", "No"],
    }
    for col in SERVICE_COLS:
        base[col] = ["Yes", "No", "Yes"]
    return pd.DataFrame(base)


def test_total_charges_blank_becomes_zero():
    df = clean_and_engineer(raw_frame())
    assert df["TotalCharges"].tolist() == [0.0, 666.0, 4320.0]


def test_target_is_binary():
    df = clean_and_engineer(raw_frame())
    assert df[TARGET].tolist() == [1, 0, 0]


def test_engineered_features_exist_and_are_row_local():
    df = clean_and_engineer(raw_frame())
    assert df.loc[1, "avg_monthly_spend"] == 55.5  # 666 / 12
    assert df.loc[1, "charges_drift"] == 0.0
    assert df.loc[2, "active_services"] == len(SERVICE_COLS)
    assert df.loc[2, "streaming_bundle"] == "Yes"
    assert df.loc[0, "tenure_bucket"] == "0-6m"


def test_groups_cover_only_existing_columns():
    df = clean_and_engineer(raw_frame())
    for cols in GROUPS.values():
        for col in cols:
            assert col in df.columns, col
