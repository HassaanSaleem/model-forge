"""End-to-end churn study: data -> selection funnel -> fusion network -> evaluation.

Run as a module (seeded, deterministic on CPU):

    python -m model_forge.churn.train --epochs 50 --outdir docs/churn

Writes metrics.json, selection_report.txt, feature_importance.csv, and
evaluation.png (confusion matrix + training history) into --outdir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET, load_dataset
from .model import FusionConfig, build_fusion_model
from .preprocess import GroupPreprocessor
from .selection import SelectionConfig, select_features, stage_significance

SEED = 1337

# Branch widths are sized to the encoded group dims this dataset produces (~5-12
# features per group after the funnel); on wider matrices the widths scale with
# each family's feature count.
BRANCHES = {
    "billing": [64, 32],
    "profile": [64, 32],
    "services": [128, 64],  # deepest branch: the behavioral group carries the most signal
}


def train(epochs: int = 60, batch_size: int = 128, outdir: Path = Path("docs/churn"),
          cache_path: Path = Path("data/telco.csv")) -> dict:
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks

    tf.keras.utils.set_random_seed(SEED)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    df, groups = load_dataset(cache_path)
    y = df[TARGET].values
    print(f"Loaded {len(df)} customers; churn rate {y.mean():.1%}")

    # ------------------------------------------------- selection stages 1-6
    config = SelectionConfig(drop_contains=("customerID",))
    surviving, reports = select_features(df, groups, config)
    report_lines = [str(r) for r in reports]
    print("\n".join(report_lines))

    # ------------------------------------------------------- split + encode
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=y)
    y_train, y_test = train_df[TARGET].values, test_df[TARGET].values

    encoded_train: dict[str, np.ndarray] = {}
    encoded_test: dict[str, np.ndarray] = {}
    preprocessors: dict[str, GroupPreprocessor] = {}
    importances: list[pd.DataFrame] = []

    for group, cols in surviving.items():
        pre = GroupPreprocessor(group)
        x_train = pre.fit_transform(train_df, cols)
        x_test = pre.transform(test_df)

        # ------------------------------------------- stage 7: significance
        mask, importance, report = stage_significance(
            x_train, y_train, pre.feature_names(), config, group)
        report_lines.append(str(report))
        print(report)

        encoded_train[group] = np.nan_to_num(x_train[:, mask], nan=0.0)
        encoded_test[group] = np.nan_to_num(x_test[:, mask], nan=0.0)
        preprocessors[group] = pre
        importances.append(importance)

    importance_table = (pd.concat(importances, ignore_index=True)
                        .sort_values("f_score", ascending=False)
                        .reset_index(drop=True))
    importance_table["global_rank"] = range(1, len(importance_table) + 1)
    importance_table.to_csv(outdir / "feature_importance.csv", index=False)
    (outdir / "selection_report.txt").write_text("\n".join(report_lines) + "\n")

    input_dims = {g: x.shape[1] for g, x in encoded_train.items()}
    print(f"Encoded input dims: {input_dims}")

    # ------------------------------------------------------------------ model
    model = build_fusion_model(input_dims, FusionConfig(
        branch_layers=BRANCHES, head_layers=[64, 32],
        dropout_rate=0.2, learning_rate=5e-4))
    model.summary()

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}

    history = model.fit(
        [encoded_train[g] for g in input_dims],
        y_train,
        validation_data=([encoded_test[g] for g in input_dims], y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_auc", patience=8, mode="max",
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4,
                                        mode="max", min_lr=1e-6, verbose=1),
        ],
        verbose=2,
    )

    # ------------------------------------------------------------------ eval
    prob = model.predict([encoded_test[g] for g in input_dims], verbose=0).flatten()
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "epochs_ran": len(history.history["loss"]),
        "best_val_auc": round(float(max(history.history["val_auc"])), 4),
        "input_dims": input_dims,
        "params": int(model.count_params()),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    _plot(history, confusion_matrix(y_test, pred), outdir / "evaluation.png")
    return metrics


def _plot(history, cm, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1], ["Retained", "Churned"])
    axes[0].set_yticks([0, 1], ["Retained", "Churned"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion matrix")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].plot(history.history["auc"], label="train AUC")
    axes[1].plot(history.history["val_auc"], label="val AUC")
    axes[1].plot(history.history["loss"], "--", alpha=0.6, label="train loss")
    axes[1].plot(history.history["val_loss"], "--", alpha=0.6, label="val loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Training history")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--outdir", type=Path, default=Path("docs/churn"))
    parser.add_argument("--data-cache", type=Path, default=Path("data/telco.csv"))
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.outdir, args.data_cache)


if __name__ == "__main__":
    main()
