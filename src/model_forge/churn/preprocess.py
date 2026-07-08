"""Per-group feature encoding.

Each feature group (billing / profile / services) is encoded independently into its
own dense matrix, because each group feeds its own branch of the fusion network.
Categoricals are label-encoded (mode-filled, unseen values at transform time fall
back to the fill value); numerics are median-filled and standardized. Column order
inside a group is always categoricals first, then numerics, so the encoded matrix
columns line up with `feature_names()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class GroupPreprocessor:
    """Fit/transform one named feature group into a single numeric matrix."""

    def __init__(self, group: str):
        self.group = group
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.encoders: dict[str, LabelEncoder] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.fill_values: dict[str, object] = {}
        self.vocab_sizes: dict[str, int] = {}

    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        """A string column is 'really' numeric if most non-null values convert."""
        if pd.api.types.is_numeric_dtype(series):
            return True
        converted = pd.to_numeric(series, errors="coerce")
        return len(series) > 0 and converted.notna().sum() / len(series) > 0.5

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        x = df[cols].copy()
        self.cat_cols, self.num_cols = [], []
        for col in cols:
            (self.num_cols if self._is_numeric(x[col]) else self.cat_cols).append(col)

        blocks: list[np.ndarray] = []
        for col in self.cat_cols:
            mode = x[col].mode()
            fill = mode.iloc[0] if not mode.empty else "unknown"
            self.fill_values[col] = fill
            values = x[col].fillna(fill).astype(str)
            encoder = LabelEncoder()
            blocks.append(encoder.fit_transform(values).reshape(-1, 1).astype(np.float32))
            self.encoders[col] = encoder
            self.vocab_sizes[col] = len(encoder.classes_)

        for col in self.num_cols:
            numeric = pd.to_numeric(x[col], errors="coerce")
            median = numeric.median()
            fill = median if pd.notna(median) else 0.0
            self.fill_values[col] = fill
            scaler = StandardScaler()
            blocks.append(scaler.fit_transform(
                numeric.fillna(fill).values.reshape(-1, 1)).astype(np.float32))
            self.scalers[col] = scaler

        return np.hstack(blocks) if blocks else np.zeros((len(df), 1), dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for col in self.cat_cols:
            fill = str(self.fill_values[col])
            encoder = self.encoders[col]
            values = df[col].fillna(fill).astype(str)
            known = set(encoder.classes_)
            values = values.where(values.isin(known), fill)
            if fill not in known:
                encoder.classes_ = np.append(encoder.classes_, fill)
            blocks.append(encoder.transform(values).reshape(-1, 1).astype(np.float32))

        for col in self.num_cols:
            numeric = pd.to_numeric(df[col], errors="coerce").fillna(self.fill_values[col])
            blocks.append(self.scalers[col].transform(
                numeric.values.reshape(-1, 1)).astype(np.float32))

        return np.hstack(blocks) if blocks else np.zeros((len(df), 1), dtype=np.float32)

    def feature_names(self) -> list[str]:
        return self.cat_cols + self.num_cols
