import numpy as np
import pandas as pd

from model_forge.churn.selection import (
    SelectionConfig,
    select_features,
    stage_correlation,
    stage_cross_group,
    stage_distribution,
    stage_name_filter,
    stage_significance,
    stage_sparsity,
    stage_unknown,
)

N = 400


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)  # fresh generator per call: every test sees the same frame
    signal = rng.normal(size=N)
    color = rng.choice(["red", "blue"], N, p=[0.7, 0.3])
    return pd.DataFrame({
        "mostly_zero": [0] * int(N * 0.95) + list(rng.integers(1, 5, N - int(N * 0.95))),
        "mostly_unknown": ["unknown"] * int(N * 0.8) + ["a"] * (N - int(N * 0.8)),
        "customer_id": range(N),
        "signal": signal,
        "signal_copy": signal * 2 + rng.normal(scale=0.01, size=N),
        "noise": rng.normal(size=N),
        "color": color,
        "colour_dup": color.copy(),  # a true duplicate: same distribution, same rows
        "colour_permuted": rng.permutation(color),  # same distribution, different rows
        "payment_method": rng.choice(["card", "cash"], N),
    })


def test_stage_sparsity_drops_dead_columns():
    df = frame()
    kept, report = stage_sparsity(df, list(df.columns), SelectionConfig(), "test")
    assert "mostly_zero" in report.dropped
    assert "signal" in kept


def test_stage_unknown_drops_placeholder_columns():
    df = frame()
    kept, report = stage_unknown(df, list(df.columns), SelectionConfig(), "test")
    assert "mostly_unknown" in report.dropped
    assert "color" in kept


def test_stage_name_filter_drops_by_substring():
    kept, report = stage_name_filter(
        ["customer_id", "signal"], SelectionConfig(drop_contains=("_id",)), "test")
    assert kept == ["signal"]
    assert report.dropped == ["customer_id"]


def test_stage_correlation_drops_near_duplicates():
    df = frame()
    kept, report = stage_correlation(
        df, ["signal", "signal_copy", "noise"], SelectionConfig(), "test")
    assert "signal_copy" in report.dropped
    assert {"signal", "noise"} <= set(kept)


def test_stage_distribution_drops_true_duplicates_only():
    df = frame()
    kept, report = stage_distribution(
        df, ["color", "colour_dup", "colour_permuted", "payment_method"],
        SelectionConfig(), "test")
    assert "colour_dup" in report.dropped  # row-identical copy dies
    assert "colour_permuted" in kept  # same marginal, different rows — real feature, survives
    assert "color" in kept


def test_stage_cross_group_prefers_primary_group():
    df = pd.DataFrame({
        "PaymentMethod": ["card", "cash"] * 50,
        "profile_payment_method": ["card", "cash"] * 50,
        "tenure": range(100),
    })
    kept, report = stage_cross_group(
        df, ["PaymentMethod"], ["profile_payment_method", "tenure"],
        SelectionConfig(), "test")
    assert kept == ["tenure"]
    assert report.dropped == ["profile_payment_method"]


def test_stage_significance_keeps_informative_features():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, N)
    informative = y * 3.0 + rng.normal(scale=0.5, size=N)
    x = np.column_stack([informative, rng.normal(size=N)])
    mask, importance, _ = stage_significance(
        x, y, ["informative", "pure_noise"], SelectionConfig(), "test")
    assert mask[0]
    assert importance.iloc[0]["feature"] == "informative"


def test_select_features_runs_the_full_funnel():
    df = frame()
    groups = {
        "billing": ["payment_method", "signal", "signal_copy"],
        "profile": ["color", "colour_dup", "mostly_unknown", "customer_id"],
    }
    surviving, reports = select_features(
        df, groups, SelectionConfig(drop_contains=("customer_id",)))
    assert "signal_copy" not in surviving["billing"]
    assert "customer_id" not in surviving["profile"]
    assert len(reports) >= 11  # 5 stages x 2 groups + >=1 cross-group pass
