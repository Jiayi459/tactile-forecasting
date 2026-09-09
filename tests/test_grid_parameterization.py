"""One shared model class serves 32x32 and 21x21 (Q-G): defaults stay bit-identical for AS."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from src.actionsense.tactile_map.models import build_model  # noqa: E402


@pytest.mark.parametrize("shape", [(2, 32, 32), (2, 21, 21), (1, 16, 16)])
@pytest.mark.parametrize("encoder", ["flatten", "cnn"])
def test_map_encoders_accept_any_grid(encoder, shape):
    m = build_model(encoder, horizon=10, in_shape=shape)
    mu, lv = m(torch.zeros(3, 5, *shape))
    assert mu.shape == lv.shape == (3, 10, 6)


def test_default_is_the_actionsense_shape():
    """No in_shape -> the 32x32 FLAT=2048 layer, so every existing AS call site is unchanged."""
    m = build_model("flatten", horizon=10)
    assert m.encoder.proj[1].in_features == 2048
    mu, _ = m(torch.zeros(2, 4, 2, 32, 32))
    assert mu.shape == (2, 10, 6)
    with pytest.raises(RuntimeError):
        m(torch.zeros(2, 4, 2, 21, 21))         # wrong sensor into an AS model fails loudly


def test_probgru_with_egotouch_shape():
    m = build_model("cnn", horizon=10, backbone="probgru", n_act=65, in_shape=(2, 21, 21))
    mu, lv = m(torch.zeros(3, 5, 2, 21, 21), torch.zeros(3, dtype=torch.long),
               torch.zeros(3, 6))
    assert mu.shape == lv.shape == (3, 10, 6)


def test_aggregate_ignores_in_shape():
    m = build_model("aggregate", horizon=10, in_shape=(2, 21, 21))
    mu, _ = m(torch.zeros(3, 5, 6))
    assert mu.shape == (3, 10, 6)
