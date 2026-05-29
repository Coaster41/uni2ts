import numpy as np
import torch


def normalize_for_model(
    series: np.ndarray,
    context_patches: int,
    patch_size: int,
) -> np.ndarray:
    """Apply the same per-series normalization as PackedStdScaler.

    Replicates: loc = mean(ctx), scale = std(ctx) + 1e-5 (minimum_scale),
    normalized = (series - loc) / scale, where ctx is the context window only.
    """
    ctx_len = context_patches * patch_size
    loc = series[:, :ctx_len].mean(axis=1, keepdims=True)
    scale = np.maximum(series[:, :ctx_len].std(axis=1, keepdims=True), 1e-5)
    return (series - loc) / scale


def make_batch(
    series: np.ndarray,
    patch_size: int,
    context_patches: int,
    pred_patches: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Convert raw numpy time series to a model-ready batch dict.

    Parameters
    ----------
    series : float32 [batch, time]
        Raw time series. ``time`` must equal
        ``(context_patches + pred_patches) * patch_size``.
    patch_size : int
    context_patches : int
    pred_patches : int
    device : torch.device or str

    Returns
    -------
    dict with keys:
        target          float32 [batch, n_patches, patch_size]
        observed_mask   bool    [batch, n_patches, patch_size]
        sample_id       long    [batch, n_patches]
        time_id         long    [batch, n_patches]
        variate_id      long    [batch, n_patches]
        prediction_mask bool    [batch, n_patches]
    """
    batch, time = series.shape
    n_patches = context_patches + pred_patches
    expected_time = n_patches * patch_size
    if time != expected_time:
        raise ValueError(
            f"series length {time} does not match "
            f"(context_patches + pred_patches) * patch_size = {expected_time}"
        )

    patched = series.reshape(batch, n_patches, patch_size)
    target = torch.from_numpy(np.ascontiguousarray(patched)).to(device)

    observed_mask = torch.ones(batch, n_patches, patch_size, dtype=torch.bool, device=device)

    # 1-indexed: PackedStdScaler treats sample_id == 0 as padding (no-op scaling).
    sample_id = (
        torch.arange(1, batch + 1, dtype=torch.long, device=device)
        .unsqueeze(1)
        .expand(-1, n_patches)
    )

    time_id = (
        torch.arange(n_patches, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch, -1)
    )

    variate_id = torch.zeros(batch, n_patches, dtype=torch.long, device=device)

    prediction_mask = torch.zeros(batch, n_patches, dtype=torch.bool, device=device)
    prediction_mask[:, -pred_patches:] = True

    return {
        "target": target,
        "observed_mask": observed_mask,
        "sample_id": sample_id,
        "time_id": time_id,
        "variate_id": variate_id,
        "prediction_mask": prediction_mask,
    }
