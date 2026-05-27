import numpy as np
import pytest
import torch

from lib.attn_extractor import AttentionExtractor
from lib.batch_prep import make_batch
from uni2ts.model.moiraic.module import MoiraicModule
from uni2ts.model.moiraie.module import MoiraieModule

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4
N_PATCHES = CONTEXT_PATCHES + PRED_PATCHES  # 36
SERIES_LENGTH = N_PATCHES * PATCH_SIZE  # 576
BATCH = 2
NUM_LAYERS = 2
NUM_HEADS = 1  # d_model=64, num_heads = 64 // 64 = 1

_TINY = dict(
    d_model=64,
    d_ff=128,
    num_layers=NUM_LAYERS,
    patch_size=PATCH_SIZE,
    max_seq_len=64,
    attn_dropout_p=0.0,
    dropout_p=0.0,
)


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(42)
    series = rng.standard_normal((BATCH, SERIES_LENGTH)).astype(np.float32)
    return make_batch(series, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES)


@pytest.fixture(scope="module")
def module_e():
    return MoiraieModule(**_TINY, num_predict_token=1).eval()


@pytest.fixture(scope="module")
def module_c():
    return MoiraicModule(**_TINY, num_predict_token=4).eval()


def test_moiraie_attn_extractor_keys(module_e, batch):
    extractor = AttentionExtractor(module_e)
    weights = extractor.run(batch)
    assert set(weights.keys()) == set(range(NUM_LAYERS))


def test_moiraie_attn_extractor_shapes(module_e, batch):
    extractor = AttentionExtractor(module_e)
    weights = extractor.run(batch)
    for layer_idx, w in weights.items():
        assert w.shape == (BATCH, NUM_HEADS, N_PATCHES, N_PATCHES), (
            f"Layer {layer_idx}: expected {(BATCH, NUM_HEADS, N_PATCHES, N_PATCHES)}, got {w.shape}"
        )


def test_moiraie_attn_extractor_non_negative(module_e, batch):
    extractor = AttentionExtractor(module_e)
    weights = extractor.run(batch)
    for layer_idx, w in weights.items():
        assert (w >= 0).all(), f"Layer {layer_idx} has negative attention weights"


def test_moiraie_attn_extractor_cpu_detached(module_e, batch):
    extractor = AttentionExtractor(module_e)
    weights = extractor.run(batch)
    for layer_idx, w in weights.items():
        assert w.device.type == "cpu"
        assert not w.requires_grad


def test_moiraic_attn_extractor_keys(module_c, batch):
    extractor = AttentionExtractor(module_c)
    weights = extractor.run(batch)
    assert set(weights.keys()) == set(range(NUM_LAYERS))


def test_moiraic_attn_extractor_shapes(module_c, batch):
    extractor = AttentionExtractor(module_c)
    weights = extractor.run(batch)
    for layer_idx, w in weights.items():
        assert w.shape == (BATCH, NUM_HEADS, N_PATCHES, N_PATCHES), (
            f"Layer {layer_idx}: expected {(BATCH, NUM_HEADS, N_PATCHES, N_PATCHES)}, got {w.shape}"
        )


def test_moiraic_attn_extractor_non_negative(module_c, batch):
    extractor = AttentionExtractor(module_c)
    weights = extractor.run(batch)
    for layer_idx, w in weights.items():
        assert (w >= 0).all(), f"Layer {layer_idx} has negative attention weights"


def test_moiraic_attn_extractor_causal(module_c, batch):
    extractor = AttentionExtractor(module_c)
    weights = extractor.run(batch)
    for layer_idx, w in weights.items():
        # Upper triangle (future tokens) should be zero due to causal mask
        upper = torch.triu(w[0, 0], diagonal=1)
        assert (upper == 0).all(), (
            f"Layer {layer_idx}: causal mask violated — upper triangle is non-zero"
        )
