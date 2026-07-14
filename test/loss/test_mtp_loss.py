import torch

from uni2ts.loss.packed.quantile import PackedQuantileMTPLoss


def test_reduce_loss_excludes_sample_boundary_windows():
    """Two packed samples (ids 1 and 2), npt=2, patch_size=1, current-token.

    Windows that straddle the sample boundary (positions 2,3) must be
    excluded from the average, even though they carry a huge loss value.
    """
    num_predict_token = 2
    patch_size = 1

    sample_id = torch.tensor([[1, 1, 1, 2, 2, 2]])
    variate_id = torch.zeros_like(sample_id)
    prediction_mask = torch.ones_like(sample_id, dtype=torch.bool)
    observed_mask = torch.ones(1, 6, patch_size, dtype=torch.bool)

    aligned_len = 6 - num_predict_token + 1  # 5 windows
    loss = torch.ones(1, aligned_len, num_predict_token * patch_size)
    boundary_window = 2  # covers positions (2, 3): sample 1 -> sample 2
    loss[0, boundary_window, :] = 100.0

    loss_fn = PackedQuantileMTPLoss(shift=False)
    result = loss_fn.reduce_loss(
        loss, prediction_mask, observed_mask, sample_id, variate_id
    )

    # Window (positions 2,3) straddles the boundary: only its first offset
    # (target position 2, still sample 1) is valid; the second offset
    # (target position 3, sample 2) must be excluded. 8 clean entries @ loss=1
    # + 1 valid boundary-window entry @ loss=100, over 9 active entries.
    assert torch.isclose(result, torch.tensor(108.0 / 9))


def test_reduce_loss_excludes_variate_boundary_windows():
    """Same sample, but variate changes mid-window -> also must be excluded."""
    num_predict_token = 2
    patch_size = 1

    sample_id = torch.ones(1, 6, dtype=torch.long)
    variate_id = torch.tensor([[0, 0, 0, 1, 1, 1]])
    prediction_mask = torch.ones_like(sample_id, dtype=torch.bool)
    observed_mask = torch.ones(1, 6, patch_size, dtype=torch.bool)

    aligned_len = 6 - num_predict_token + 1
    loss = torch.ones(1, aligned_len, num_predict_token * patch_size)
    boundary_window = 2
    loss[0, boundary_window, :] = 100.0

    loss_fn = PackedQuantileMTPLoss(shift=False)
    result = loss_fn.reduce_loss(
        loss, prediction_mask, observed_mask, sample_id, variate_id
    )

    assert torch.isclose(result, torch.tensor(108.0 / 9))


def test_reduce_loss_default_unclamped():
    """FIX A: by default (clamp_loss=None) losses contribute at full value,
    unlike the old hardcoded clamp_loss=3."""
    num_predict_token = 1
    patch_size = 1

    sample_id = torch.ones(1, 4, dtype=torch.long)
    variate_id = torch.zeros_like(sample_id)
    prediction_mask = torch.ones_like(sample_id, dtype=torch.bool)
    observed_mask = torch.ones(1, 4, patch_size, dtype=torch.bool)

    loss = torch.tensor([[[10.0], [10.0], [10.0], [10.0]]])

    loss_fn = PackedQuantileMTPLoss(shift=False)
    result = loss_fn.reduce_loss(
        loss, prediction_mask, observed_mask, sample_id, variate_id
    )

    assert torch.isclose(result, torch.tensor(10.0))
    assert loss_fn.clamp_loss is None


def test_reduce_loss_configurable_clamp():
    """clamp_loss, when set, caps each token's loss before averaging."""
    num_predict_token = 1
    patch_size = 1

    sample_id = torch.ones(1, 4, dtype=torch.long)
    variate_id = torch.zeros_like(sample_id)
    prediction_mask = torch.ones_like(sample_id, dtype=torch.bool)
    observed_mask = torch.ones(1, 4, patch_size, dtype=torch.bool)

    loss = torch.tensor([[[10.0], [10.0], [10.0], [10.0]]])

    loss_fn = PackedQuantileMTPLoss(shift=False, clamp_loss=5.0)
    result = loss_fn.reduce_loss(
        loss, prediction_mask, observed_mask, sample_id, variate_id
    )

    assert torch.isclose(result, torch.tensor(5.0))
