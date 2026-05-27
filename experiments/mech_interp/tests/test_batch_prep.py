import numpy as np
import pytest
import torch
from einops import rearrange

from lib.batch_prep import make_batch
from uni2ts.model.moiraic.forecast import MoiraicForecast
from uni2ts.model.moiraic.module import MoiraicModule
from uni2ts.model.moiraie.forecast import MoiraieForecast
from uni2ts.model.moiraie.module import MoiraieModule

PATCH_SIZE = 16
CONTEXT_PATCHES = 32
PRED_PATCHES = 4
N_PATCHES = CONTEXT_PATCHES + PRED_PATCHES  # 36
SERIES_LENGTH = N_PATCHES * PATCH_SIZE  # 576
BATCH = 3

_TINY_KWARGS = dict(
    d_model=64,
    d_ff=128,
    num_layers=2,
    patch_size=PATCH_SIZE,
    max_seq_len=64,
    attn_dropout_p=0.0,
    dropout_p=0.0,
)


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    series = rng.standard_normal((BATCH, SERIES_LENGTH)).astype(np.float32)
    return make_batch(series, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES)


def test_make_batch_shapes(batch):
    assert batch["target"].shape == (BATCH, N_PATCHES, PATCH_SIZE)
    assert batch["observed_mask"].shape == (BATCH, N_PATCHES, PATCH_SIZE)
    assert batch["sample_id"].shape == (BATCH, N_PATCHES)
    assert batch["time_id"].shape == (BATCH, N_PATCHES)
    assert batch["variate_id"].shape == (BATCH, N_PATCHES)
    assert batch["prediction_mask"].shape == (BATCH, N_PATCHES)


def test_make_batch_values(batch):
    assert batch["observed_mask"].all()
    assert not batch["variate_id"].any()
    assert not batch["prediction_mask"][:, :CONTEXT_PATCHES].any()
    assert batch["prediction_mask"][:, CONTEXT_PATCHES:].all()
    assert torch.equal(batch["time_id"][0], torch.arange(N_PATCHES))
    for i in range(BATCH):
        assert batch["sample_id"][i].eq(i + 1).all(), f"sample_id row {i} should be all {i+1}"


def test_make_batch_target_patchification():
    series = np.arange(SERIES_LENGTH * BATCH, dtype=np.float32).reshape(BATCH, SERIES_LENGTH)
    b = make_batch(series, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES)
    target = b["target"]
    for i in range(BATCH):
        for j in range(N_PATCHES):
            for k in range(PATCH_SIZE):
                expected = series[i, j * PATCH_SIZE + k]
                assert target[i, j, k].item() == pytest.approx(expected), (
                    f"target[{i},{j},{k}] mismatch"
                )


def test_make_batch_moiraie_forward(batch):
    module = MoiraieModule(**_TINY_KWARGS, num_predict_token=1).eval()
    with torch.no_grad():
        out = module(**batch, training_mode=False)
    num_quantiles = len(module.quantile_levels)  # 9
    expected_last = 1 * num_quantiles * PATCH_SIZE  # 144
    assert out.shape == (BATCH, N_PATCHES, expected_last)


def test_make_batch_moiraic_forward(batch):
    module = MoiraicModule(**_TINY_KWARGS, num_predict_token=4).eval()
    with torch.no_grad():
        out = module(**batch, training_mode=False, past_cache=None, return_cache=False)
    num_quantiles = len(module.quantile_levels)  # 9
    expected_last = 4 * num_quantiles * PATCH_SIZE  # 576
    assert out.shape == (BATCH, N_PATCHES, expected_last)


def test_make_batch_matches_moiraie_forecast():
    """
    make_batch + MoiraieModule.forward() is numerically identical to MoiraieForecast.forward().

    _convert() produces the same tensors as make_batch() for a clean, exact-length series:
    - time_id: _generate_time_id yields [0..35] for a fully-observed 512-step context
    - sample_id: all-1s (single unpadded example, no padding tokens)
    Values at prediction-window positions differ (make_batch uses actual series; _convert uses
    zeros), but those tokens are replaced by mask_encoding inside the module, so they have no
    effect. The Forecast output [1, Q, 64] is a reshape of raw preds[:, 32:36, :].
    """
    rng = np.random.default_rng(1)
    series = rng.standard_normal((1, SERIES_LENGTH)).astype(np.float32)

    module = MoiraieModule(**_TINY_KWARGS, num_predict_token=1).eval()
    forecast = MoiraieForecast(
        module=module,
        prediction_length=PRED_PATCHES * PATCH_SIZE,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=CONTEXT_PATCHES * PATCH_SIZE,
    ).eval()

    context = torch.from_numpy(series[:, : CONTEXT_PATCHES * PATCH_SIZE]).unsqueeze(-1)
    past_observed = torch.ones(1, CONTEXT_PATCHES * PATCH_SIZE, 1, dtype=torch.bool)
    past_is_pad = torch.zeros(1, CONTEXT_PATCHES * PATCH_SIZE, dtype=torch.long)

    with torch.no_grad():
        forecast_out = forecast.forward(context, past_observed, past_is_pad)
        # [1, Q, prediction_length]

        batch = make_batch(series, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES)
        raw = module(**batch, training_mode=False)
        # [1, 36, Q * patch_size]; moiraie extracts prediction tokens [32..35]
        pred_raw = raw[:, CONTEXT_PATCHES:, :]  # [1, 4, Q*P]
        from_raw = rearrange(
            pred_raw,
            "b seq (q p) -> b q (seq p)",
            q=module.num_quantiles,
            p=PATCH_SIZE,
        )  # [1, Q, 64]

    assert from_raw.shape == forecast_out.shape
    assert torch.allclose(from_raw, forecast_out), (
        f"max diff: {(from_raw - forecast_out).abs().max().item()}"
    )


def test_make_batch_matches_moiraic_forecast():
    """
    make_batch + MoiraicModule.forward() is numerically identical to MoiraicForecast.forward()
    in the non-AR (single-shot) case, which applies when pred_patches == num_predict_token.

    MoiraicForecast uses pred_index = [context_patches - 1] = [31]: the last context token
    predicts all num_predict_token future patches at once. will_ar is False because
    pred_patches (4) == num_predict_token (4).  The Forecast output [1, Q, 64] is a reshape
    of raw preds[:, 31, :] (the last context token's output).
    """
    rng = np.random.default_rng(2)
    series = rng.standard_normal((1, SERIES_LENGTH)).astype(np.float32)

    module = MoiraicModule(**_TINY_KWARGS, num_predict_token=PRED_PATCHES).eval()
    forecast = MoiraicForecast(
        module=module,
        prediction_length=PRED_PATCHES * PATCH_SIZE,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=CONTEXT_PATCHES * PATCH_SIZE,
        ar_method=None,
    ).eval()

    context = torch.from_numpy(series[:, : CONTEXT_PATCHES * PATCH_SIZE]).unsqueeze(-1)
    past_observed = torch.ones(1, CONTEXT_PATCHES * PATCH_SIZE, 1, dtype=torch.bool)
    past_is_pad = torch.zeros(1, CONTEXT_PATCHES * PATCH_SIZE, dtype=torch.long)

    with torch.no_grad():
        forecast_out = forecast.forward(context, past_observed, past_is_pad)
        # [1, Q, prediction_length]

        batch = make_batch(series, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES)
        raw = module(**batch, training_mode=False, past_cache=None, return_cache=False)
        # [1, 36, num_predict_token * Q * P]; moiraic reads prediction from last context token
        last_ctx = raw[:, CONTEXT_PATCHES - 1, :]  # [1, num_predict_token * Q * P]
        from_raw = rearrange(
            last_ctx,
            "b (t q p) -> b q (t p)",
            t=PRED_PATCHES,
            q=module.num_quantiles,
            p=PATCH_SIZE,
        )  # [1, Q, 64]

    assert from_raw.shape == forecast_out.shape
    assert torch.allclose(from_raw, forecast_out), (
        f"max diff: {(from_raw - forecast_out).abs().max().item()}"
    )


def test_make_batch_invalid_length():
    series = np.zeros((BATCH, SERIES_LENGTH + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="series length"):
        make_batch(series, PATCH_SIZE, CONTEXT_PATCHES, PRED_PATCHES)
