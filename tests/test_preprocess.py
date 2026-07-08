import numpy as np
import pandas as pd

from model_forge.churn.preprocess import GroupPreprocessor


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "contract": ["month", "year", "month", None],
        "charges": [10.0, 20.0, None, 40.0],
        "stringy_number": ["1", "2", "3", "bad"],
    })


def test_fit_transform_shape_and_order():
    pre = GroupPreprocessor("billing")
    x = pre.fit_transform(frame(), ["contract", "charges", "stringy_number"])
    assert x.shape == (4, 3)
    # categoricals first, then numerics — stringy_number is >50% numeric so it's numeric
    assert pre.feature_names() == ["contract", "charges", "stringy_number"]
    assert pre.cat_cols == ["contract"]
    assert set(pre.num_cols) == {"charges", "stringy_number"}


def test_numeric_columns_are_standardized():
    pre = GroupPreprocessor("billing")
    x = pre.fit_transform(frame(), ["charges"])
    assert abs(float(x.mean())) < 1e-6  # standardized around 0


def test_transform_handles_unseen_categories():
    pre = GroupPreprocessor("billing")
    pre.fit_transform(frame(), ["contract"])
    unseen = pd.DataFrame({"contract": ["biennial", "month"]})
    x = pre.transform(unseen)
    assert x.shape == (2, 1)
    assert not np.isnan(x).any()


def test_empty_column_list_yields_placeholder():
    pre = GroupPreprocessor("empty")
    x = pre.fit_transform(frame(), [])
    assert x.shape == (4, 1)
    assert (x == 0).all()
