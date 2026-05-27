import numpy as np
import pytest
import torch

from uni2ts.model.moiraic.module import MoiraicModule
from uni2ts.model.moiraie.module import MoiraieModule

from experiments.mech_interp.lib import ResidualExtractor, make_batch

_TINY = dict(
    d_model=64,
    d_ff=128,
    num_layers=2,
    patch_size=16,
    max_seq_len=64,
    attn_dropout_p=0.0,
    dropout_p=0.0,
)

BATCH_SIZE = 3
SERIES_LENGTH = 576  # (32 + 4) * 16
PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4
N_PATCHES = CONTEXT_PATCHES + PRED_PATCHES  # 36


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return rng.standard_normal((BATCH_SIZE, SERIES_LENGTH)).astype(np.float32)


@pytest.fixture
def batch(series):
    return make_batch(series, patch_size=PATCH_SIZE, context_patches=CONTEXT_PATCHES, pred_patches=PRED_PATCHES)


@pytest.fixture
def module_e():
    return MoiraieModule(**_TINY, num_predict_token=1).eval()


@pytest.fixture
def module_c():
    return MoiraicModule(**_TINY, num_predict_token=4).eval()


def test_residual_extractor_moiraie_shapes(module_e, batch):
    with ResidualExtractor(module_e) as extractor:
        acts = extractor.run(batch)

    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(acts.keys()) == expected_keys
    for layer_idx, tensor in acts.items():
        assert tensor.shape == (BATCH_SIZE, N_PATCHES, _TINY["d_model"]), (
            f"Layer {layer_idx}: expected {(BATCH_SIZE, N_PATCHES, _TINY['d_model'])}, got {tensor.shape}"
        )


def test_residual_extractor_moiraic_shapes(module_c, batch):
    with ResidualExtractor(module_c) as extractor:
        acts = extractor.run(batch)

    expected_keys = {-1} | set(range(_TINY["num_layers"]))
    assert set(acts.keys()) == expected_keys
    for layer_idx, tensor in acts.items():
        assert tensor.shape == (BATCH_SIZE, N_PATCHES, _TINY["d_model"]), (
            f"Layer {layer_idx}: expected {(BATCH_SIZE, N_PATCHES, _TINY['d_model'])}, got {tensor.shape}"
        )


def test_residual_extractor_detached_cpu(module_e, batch):
    with ResidualExtractor(module_e) as extractor:
        acts = extractor.run(batch)

    for layer_idx, tensor in acts.items():
        assert tensor.device == torch.device("cpu"), f"Layer {layer_idx} tensor not on CPU"
        assert tensor.grad_fn is None, f"Layer {layer_idx} tensor still has grad_fn"


def test_residual_extractor_context_manager_cleanup(module_e, batch):
    hooks_before = sum(len(layer._forward_hooks) for layer in module_e.encoder.layers)
    in_proj_hooks_before = len(module_e.in_proj._forward_hooks)

    with ResidualExtractor(module_e) as extractor:
        hooks_during = sum(len(layer._forward_hooks) for layer in module_e.encoder.layers)
        in_proj_hooks_during = len(module_e.in_proj._forward_hooks)
        extractor.run(batch)

    hooks_after = sum(len(layer._forward_hooks) for layer in module_e.encoder.layers)
    in_proj_hooks_after = len(module_e.in_proj._forward_hooks)

    assert hooks_during == hooks_before + _TINY["num_layers"]
    assert in_proj_hooks_during == in_proj_hooks_before + 1
    assert hooks_after == hooks_before
    assert in_proj_hooks_after == in_proj_hooks_before


def test_residual_extractor_run_outside_context_raises(module_e, batch):
    extractor = ResidualExtractor(module_e)
    with pytest.raises(RuntimeError):
        extractor.run(batch)


def test_residual_extractor_no_grad_pollution(module_e, batch):
    with ResidualExtractor(module_e) as extractor:
        extractor.run(batch)

    for p in module_e.parameters():
        assert p.grad is None
