"""Seven-stage feature-selection funnel.

The funnel runs in two halves. Stages 1-3 are cheap row-count statistics that a
warehouse can compute (they can run as plain SQL inside the warehouse so that dead
columns never leave it); stages 4-7
need the actual value distributions and run in pandas. Every stage returns the
surviving column list plus a small report row, so the whole funnel is auditable:
you can print exactly which stage killed which feature and why.

Stages
    1. sparsity        — drop columns that are null/zero/empty in > max_zero_ratio of rows
    2. unknown ratio   — drop categoricals whose value is 'unknown' in > max_unknown_ratio
    3. name filter     — drop columns whose name contains a configured substring
    4. correlation     — drop numeric columns correlated > correlation_threshold with a survivor
    5. distribution    — drop categoricals whose value distribution is near-identical to a survivor
    6. cross-group     — drop group-B columns that duplicate group-A columns (name tokens
                         or distributions), so e.g. a profile copy of a billing field dies
    7. significance    — keep only features with f_classif p-value < p_threshold vs the target
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SelectionConfig:
    max_zero_ratio: float = 0.90
    max_unknown_ratio: float = 0.50
    drop_contains: tuple[str, ...] = ("_id", "ID")
    correlation_threshold: float = 0.90
    distribution_similarity: float = 0.95
    p_threshold: float = 0.05


@dataclass
class StageReport:
    stage: str
    group: str
    dropped: list[str] = field(default_factory=list)
    kept: int = 0

    def __str__(self) -> str:
        head = f"[{self.group}] {self.stage}: dropped {len(self.dropped)}, kept {self.kept}"
        if self.dropped:
            head += f" (e.g. {self.dropped[:4]})"
        return head


# --------------------------------------------------------------------------- #
# Stages 1-3: row-count statistics (warehouse-computable)
# --------------------------------------------------------------------------- #

def stage_sparsity(df: pd.DataFrame, cols: list[str], config: SelectionConfig,
                   group: str) -> tuple[list[str], StageReport]:
    """Drop columns that are null, zero, or empty in more than max_zero_ratio of rows."""
    dropped = []
    for col in cols:
        s = df[col]
        dead = s.isna() | (s.astype(str).str.strip().isin(["", "0", "0.0", "null", "None"]))
        if dead.mean() > config.max_zero_ratio:
            dropped.append(col)
    kept = [c for c in cols if c not in dropped]
    return kept, StageReport("1 sparsity", group, dropped, len(kept))


def _is_categorical(s: pd.Series) -> bool:
    """Anything non-numeric is treated as categorical (object in pandas 2, str in pandas 3)."""
    return not pd.api.types.is_numeric_dtype(s)


def stage_unknown(df: pd.DataFrame, cols: list[str], config: SelectionConfig,
                  group: str) -> tuple[list[str], StageReport]:
    """Drop categorical columns dominated by an 'unknown' placeholder."""
    dropped = []
    for col in cols:
        s = df[col]
        if _is_categorical(s):
            unknown = s.fillna("unknown").astype(str).str.lower().eq("unknown")
            if unknown.mean() > config.max_unknown_ratio:
                dropped.append(col)
    kept = [c for c in cols if c not in dropped]
    return kept, StageReport("2 unknown", group, dropped, len(kept))


def stage_name_filter(cols: list[str], config: SelectionConfig,
                      group: str) -> tuple[list[str], StageReport]:
    """Drop columns whose name contains any configured substring (ids, dates, debug fields)."""
    dropped = [c for c in cols if any(s in c for s in config.drop_contains)]
    kept = [c for c in cols if c not in dropped]
    return kept, StageReport("3 name filter", group, dropped, len(kept))


# --------------------------------------------------------------------------- #
# Stages 4-5: value distributions (pandas-side)
# --------------------------------------------------------------------------- #

def stage_correlation(df: pd.DataFrame, cols: list[str], config: SelectionConfig,
                      group: str) -> tuple[list[str], StageReport]:
    """Drop numeric columns that are highly correlated with an earlier survivor."""
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    dropped: list[str] = []
    if len(numeric) > 1:
        corr = df[numeric].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        dropped = [c for c in upper.columns if (upper[c] > config.correlation_threshold).any()]
    kept = [c for c in cols if c not in dropped]
    return kept, StageReport("4 correlation", group, dropped, len(kept))


def _value_distribution(s: pd.Series) -> pd.Series:
    return s.fillna("_null_").astype(str).value_counts(normalize=True)


def _distributions_similar(a: pd.Series, b: pd.Series, threshold: float) -> bool:
    """Two categoricals are near-identical if they share most of their support and
    their frequencies over the shared support differ by less than 1 - threshold."""
    common = a.index.intersection(b.index)
    if len(common) == 0 or len(common) < max(len(a), len(b)) * 0.8:
        return False
    similarity = 1 - np.mean(np.abs(a.reindex(common, fill_value=0).values
                                    - b.reindex(common, fill_value=0).values))
    return similarity > threshold


def _row_agreement(a: pd.Series, b: pd.Series) -> float:
    return float((a.fillna("_null_").astype(str).values
                  == b.fillna("_null_").astype(str).values).mean())


def _duplicate_categoricals(a: pd.Series, b: pd.Series, threshold: float) -> bool:
    """A column duplicates another only if the marginal distributions match AND the
    rows actually agree. On a wide pivoted matrix the marginal test alone is a safe
    proxy (thousands of distinct value distributions make collisions unlikely), but
    on a narrow table two genuinely different Yes/No features can share a marginal
    by coincidence — the row-agreement check keeps those alive."""
    return (_distributions_similar(_value_distribution(a), _value_distribution(b), threshold)
            and _row_agreement(a, b) > threshold)


def stage_distribution(df: pd.DataFrame, cols: list[str], config: SelectionConfig,
                       group: str) -> tuple[list[str], StageReport]:
    """Drop categoricals that duplicate an earlier survivor (marginals + row agreement)."""
    cats = [c for c in cols if _is_categorical(df[c])]
    dropped: set[str] = set()
    for i, a in enumerate(cats):
        if a in dropped:
            continue
        for b in cats[i + 1:]:
            if b in dropped:
                continue
            if _duplicate_categoricals(df[a], df[b], config.distribution_similarity):
                dropped.add(b)
    kept = [c for c in cols if c not in dropped]
    return kept, StageReport("5 distribution", group, sorted(dropped), len(kept))


# --------------------------------------------------------------------------- #
# Stage 6: cross-group dedup
# --------------------------------------------------------------------------- #

_NOISE_TOKENS = {"profile", "invoice", "billing", "current", "last", "first", "total", "is", ""}


def _name_tokens(col: str) -> set[str]:
    """Normalize a column name to comparable tokens.

    'profile_current_payment_method' -> {'current', 'payment', 'method'}
    'PaymentMethod'                  -> {'payment', 'method'}

    camelCase splits before lowercasing — lowering first would fuse 'PaymentMethod'
    into one unsplittable token.
    """
    name = re.sub(r"^profile_", "", col, flags=re.IGNORECASE)
    if "__" in name:  # event-style names: Event__Property__value__agg -> property part
        parts = name.split("__")
        name = parts[1] if len(parts) > 1 else parts[0]
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return set(re.split(r"[_\s]+", name.lower())) - {""}


def _names_match(col_a: str, col_b: str) -> bool:
    """Same concept if one name's core tokens subset the other's, or they overlap >= 70%."""
    core_a = _name_tokens(col_a) - _NOISE_TOKENS or _name_tokens(col_a)
    core_b = _name_tokens(col_b) - _NOISE_TOKENS or _name_tokens(col_b)
    if not core_a or not core_b:
        return False
    if core_a.issubset(core_b) or core_b.issubset(core_a):
        return True
    overlap = len(core_a & core_b)
    return overlap / min(len(core_a), len(core_b)) >= 0.7


def stage_cross_group(df: pd.DataFrame, primary_cols: list[str], secondary_cols: list[str],
                      config: SelectionConfig, group: str) -> tuple[list[str], StageReport]:
    """Drop secondary-group columns that duplicate a primary-group column.

    The primary group wins: if 'PaymentMethod' lives in billing, a profile-group
    copy of the same concept (matched by name tokens, confirmed or caught by
    distribution similarity for categoricals) is removed from the secondary group.
    """
    dropped: set[str] = set()
    for col_b in secondary_cols:
        for col_a in primary_cols:
            if _names_match(col_a, col_b):
                dropped.add(col_b)
                break
            if _is_categorical(df[col_a]) and _is_categorical(df[col_b]):
                if _duplicate_categoricals(df[col_a], df[col_b],
                                           config.distribution_similarity):
                    dropped.add(col_b)
                    break
    kept = [c for c in secondary_cols if c not in dropped]
    return kept, StageReport("6 cross-group", group, sorted(dropped), len(kept))


# --------------------------------------------------------------------------- #
# Stage 7: statistical significance (post-encoding)
# --------------------------------------------------------------------------- #

def stage_significance(x: np.ndarray, y: np.ndarray, feature_names: list[str],
                       config: SelectionConfig, group: str
                       ) -> tuple[np.ndarray, pd.DataFrame, StageReport]:
    """Keep only features whose f_classif p-value beats p_threshold.

    Runs on the *encoded* train matrix (never test — that would leak), and returns
    the boolean keep-mask, a ranked importance table, and the stage report.
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    if x.shape[1] == 0:
        return np.array([], dtype=bool), pd.DataFrame(), StageReport("7 significance", group, [], 0)

    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(x, y)
    importance = (pd.DataFrame({
        "feature": feature_names,
        "f_score": selector.scores_,
        "p_value": selector.pvalues_,
        "group": group,
    }).sort_values("f_score", ascending=False).reset_index(drop=True))
    importance["rank"] = range(1, len(importance) + 1)

    mask = np.asarray(importance.set_index("feature").loc[feature_names, "p_value"]
                      < config.p_threshold)
    dropped = [f for f, keep in zip(feature_names, mask, strict=True) if not keep]
    return mask, importance, StageReport("7 significance", group, dropped, int(mask.sum()))


# --------------------------------------------------------------------------- #
# The funnel
# --------------------------------------------------------------------------- #

def select_within_group(df: pd.DataFrame, cols: list[str], config: SelectionConfig,
                        group: str) -> tuple[list[str], list[StageReport]]:
    """Run stages 1-5 for one feature group, collecting per-stage reports."""
    reports: list[StageReport] = []
    kept, report = stage_sparsity(df, cols, config, group)
    reports.append(report)
    kept, report = stage_unknown(df, kept, config, group)
    reports.append(report)
    kept, report = stage_name_filter(kept, config, group)
    reports.append(report)
    kept, report = stage_correlation(df, kept, config, group)
    reports.append(report)
    kept, report = stage_distribution(df, kept, config, group)
    reports.append(report)
    return kept, reports


def select_features(df: pd.DataFrame, groups: dict[str, list[str]],
                    config: SelectionConfig | None = None,
                    ) -> tuple[dict[str, list[str]], list[StageReport]]:
    """Run stages 1-6 across ordered feature groups.

    Groups are ordered by priority: the first group is most authoritative, so in
    stage 6 later groups lose their duplicates of earlier groups' concepts.
    Stage 7 runs separately (it needs encoded matrices) — see stage_significance.
    """
    config = config or SelectionConfig()
    reports: list[StageReport] = []
    surviving: dict[str, list[str]] = {}

    for group, cols in groups.items():
        kept, group_reports = select_within_group(df, cols, config, group)
        surviving[group] = kept
        reports.extend(group_reports)

    names = list(surviving)
    for i, primary in enumerate(names):
        for secondary in names[i + 1:]:
            kept, report = stage_cross_group(
                df, surviving[primary], surviving[secondary], config,
                f"{secondary} vs {primary}")
            surviving[secondary] = kept
            reports.append(report)

    return surviving, reports
