"""Differentiable forecaster interface for the white-box attack path.

The block2 ``ForecastAdapter`` returns numpy and runs under ``torch.no_grad``;
it is useless for gradient attacks. A ``GradForecaster`` is its differentiable
twin: ``[n, ctx] -> [n, Q, H]`` with the autograd graph intact back to the raw
context tensor. Model weights are frozen (``requires_grad_(False)``) — the only
thing carrying gradient is the input.

Everything here is single-forward-pass by construction. Causal / next-token
models emit at most ``num_predict_token * patch_size`` steps from the last
context patch, so a horizon beyond that would require an AR unroll; we raise
instead (see HANDOFF §7).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)
MEDIAN_IDX = 4


def make_batch_torch(
    series: torch.Tensor,
    patch_size: int,
    context_patches: int,
    pred_patches: int,
) -> dict[str, torch.Tensor]:
    """Torch/differentiable twin of ``lib.batch_prep.make_batch``.

    ``series`` is ``[B, T]`` and stays in the autograd graph — no ``.detach()``,
    no numpy round-trip. Every other tensor is a constant index tensor.
    """
    B, T = series.shape
    n_patches = context_patches + pred_patches
    if T != n_patches * patch_size:
        raise ValueError(
            f"series length {T} != (context_patches + pred_patches) * patch_size "
            f"= {n_patches * patch_size}"
        )
    dev = series.device
    target = series.reshape(B, n_patches, patch_size)  # grad flows through here
    observed_mask = torch.ones(
        B, n_patches, patch_size, dtype=torch.bool, device=dev
    )
    # 1-indexed: PackedStdScaler treats sample_id == 0 as padding and would
    # silently skip the scaling.
    sample_id = (
        torch.arange(1, B + 1, dtype=torch.long, device=dev)
        .unsqueeze(1)
        .expand(-1, n_patches)
    )
    time_id = (
        torch.arange(n_patches, dtype=torch.long, device=dev)
        .unsqueeze(0)
        .expand(B, -1)
    )
    variate_id = torch.zeros(B, n_patches, dtype=torch.long, device=dev)
    prediction_mask = torch.zeros(B, n_patches, dtype=torch.bool, device=dev)
    prediction_mask[:, -pred_patches:] = True
    return {
        "target": target,
        "observed_mask": observed_mask,
        "sample_id": sample_id,
        "time_id": time_id,
        "variate_id": variate_id,
        "prediction_mask": prediction_mask,
    }


class GradForecaster(ABC):
    """Differentiable ``[n, ctx] -> [n, Q, H]`` forecaster with frozen weights."""

    patch_size: int
    npt: int
    num_quantiles: int
    predict_next: bool
    device: str
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILE_LEVELS

    @abstractmethod
    def quantiles(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        """Quantile forecast ``[n, Q, horizon]``, differentiable wrt ``context``."""
        raise NotImplementedError

    def median(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        """Median forecast ``[n, horizon]``."""
        return self.quantiles(context, horizon)[:, self.median_idx, :]

    @property
    def median_idx(self) -> int:
        return min(
            range(len(self.quantile_levels)),
            key=lambda i: abs(self.quantile_levels[i] - 0.5),
        )

    @property
    def single_pass_cap(self) -> int:
        """Max horizon reachable in one forward pass."""
        return self.npt * self.patch_size

    def _check_geometry(self, ctx: int, horizon: int) -> tuple[int, int]:
        P = self.patch_size
        if ctx % P != 0:
            raise ValueError(f"context length {ctx} not divisible by patch_size {P}")
        if horizon % P != 0:
            raise ValueError(f"horizon {horizon} not divisible by patch_size {P}")
        if self.predict_next and horizon > self.single_pass_cap:
            raise ValueError(
                f"horizon {horizon} exceeds single-pass capacity "
                f"{self.single_pass_cap} for a next-token model; "
                "AR unroll is not implemented (see HANDOFF §7)."
            )
        return ctx // P, horizon // P

    def quantiles_batched(
        self, context: torch.Tensor, horizon: int, batch_size: int = 64
    ) -> torch.Tensor:
        """Chunked, no-grad forward. For clean metrics only, never for attacks."""
        outs = []
        with torch.no_grad():
            for i in range(0, context.shape[0], batch_size):
                outs.append(self.quantiles(context[i : i + batch_size], horizon))
        return torch.cat(outs, dim=0)
