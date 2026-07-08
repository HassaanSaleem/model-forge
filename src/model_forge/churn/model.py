"""Multi-input fusion network.

One dense branch per feature group, then a concatenation ("fusion") layer, then a
classification head. The point of the architecture is that each branch gets a
capacity budget matched to its group's signal density — a wide/deep branch for the
high-dimensional behavioral group, smaller branches for compact billing/profile
groups — instead of one tower that lets the largest group drown out the others.

TensorFlow imports are deferred to call time so that importing this package (and
running the non-model tests) never pays the TF startup cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionConfig:
    branch_layers: dict[str, list[int]] = field(default_factory=dict)
    head_layers: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.25
    learning_rate: float = 1e-3


def build_fusion_model(input_dims: dict[str, int], config: FusionConfig):
    """Build and compile the fusion network.

    input_dims maps group name -> encoded feature count; config.branch_layers maps
    group name -> that branch's dense-layer widths (defaults to [256, 128, 64]).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    def branch(input_layer, widths: list[int], name: str):
        x = input_layer
        for i, width in enumerate(widths):
            x = layers.Dense(width, name=f"{name}_dense{i + 1}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(config.dropout_rate)(x)
        return x

    inputs, branch_outputs = [], []
    for group, dim in input_dims.items():
        input_layer = layers.Input(shape=(dim,), name=f"{group}_input")
        inputs.append(input_layer)
        widths = config.branch_layers.get(group, [256, 128, 64])
        branch_outputs.append(branch(input_layer, widths, group))

    merged = layers.Concatenate(name="fusion")(branch_outputs)

    x = merged
    for i, width in enumerate(config.head_layers):
        x = layers.Dense(width, name=f"head_dense{i + 1}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(config.dropout_rate)(x)

    output = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs=inputs, outputs=output, name="churn_fusion")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
