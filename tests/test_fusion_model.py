import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from model_forge.churn.model import FusionConfig, build_fusion_model  # noqa: E402


def test_fusion_model_shapes_and_compile():
    dims = {"billing": 5, "profile": 3, "services": 8}
    model = build_fusion_model(
        dims, FusionConfig(branch_layers={"billing": [16], "profile": [16], "services": [32, 16]},
                           head_layers=[16]))
    assert model.count_params() > 0
    assert [inp.name for inp in model.inputs] == [
        "billing_input", "profile_input", "services_input"]

    batch = [np.zeros((4, d), dtype=np.float32) for d in dims.values()]
    out = model.predict(batch, verbose=0)
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()  # sigmoid output
