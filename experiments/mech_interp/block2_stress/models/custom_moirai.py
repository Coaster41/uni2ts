"""
Adapters for the custom moiraic / moiraie models.

These wrap the existing block1 probing path (``lib.utils._load_module`` +
``lib.batch_prep.make_batch``) and the moirai-specific quantile reshaping that
used to live in ``run_forecasts._extract_fq``. The raw-module path itself is
untouched — these adapters only *reuse* it — so block1 mech-interp work is
unaffected.

All torch / uni2ts imports are deferred to construction time so importing the
adapter registry stays cheap in environments without torch.

Numerical note: the stress battery only hands us the context window. We rebuild
a full model input by zero-padding the prediction region. This is exactly
equivalent to passing the real future values, because moiraie overwrites the
prediction-region representations with a learned mask embedding
(``mask_fill`` in moiraie/module.py) and moiraic is causal and read at the last
context token — neither output depends on the prediction-region target.
"""
from __future__ import annotations

import numpy as np

from .base import DEFAULT_QUANTILE_LEVELS, ForecastAdapter


def _extract_fq(
    result: np.ndarray,
    is_moiraic: bool,
    npt: int,
    Q: int,
    P: int,
    context_patches: int,
    pred_patches: int,
) -> np.ndarray:
    """result: [B, n_patches, npt*Q*P] -> [B, Q, pred_patches*P]."""
    B = result.shape[0]
    if is_moiraic:
        pred = result[:, context_patches - 1, :]  # [B, npt*Q*P]
        pred = pred.reshape(B, npt, Q, P)  # [B, npt, Q, P]
        pred = pred.transpose(0, 2, 1, 3)  # [B, Q, npt, P]
        pred = pred.reshape(B, Q, -1)[:, :, : pred_patches * P]
    else:
        pred = result[:, context_patches:, :]  # [B, pred_patches, npt*Q*P]
        pred = pred.reshape(B, pred_patches, npt, Q, P)
        pred = pred[:, :, 0, :, :]  # [B, pred_patches, Q, P]
        pred = pred.transpose(0, 2, 1, 3)  # [B, Q, pred_patches, P]
        pred = pred.reshape(B, Q, pred_patches * P)  # [B, Q, H]
    return pred


class _CustomMoiraiAdapter(ForecastAdapter):
    """Shared implementation for the encoder / decoder custom models."""

    model_name: str  # "moiraic" or "moiraie", set by subclass

    def __init__(self, ckpt: str | None = None, device: str = "cpu", **_ignored):
        # Lazy: torch + uni2ts only needed for the custom models.
        import torch  # noqa: F401

        from experiments.mech_interp.lib.utils import _load_module

        self.device = device
        self.module = _load_module(ckpt, self.model_name, device)
        self.module.eval()
        self.patch_size = int(self.module.patch_size)
        self.num_predict_token = int(self.module.num_predict_token)
        # Custom models already emit the canonical decile grid.
        self.quantile_levels = tuple(
            getattr(self.module, "quantile_levels", DEFAULT_QUANTILE_LEVELS)
        )
        if len(self.quantile_levels) != int(self.module.num_quantiles):
            # Fall back to a plain index range if the module only exposes a count.
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch

        from experiments.mech_interp.lib import make_batch

        context = np.asarray(context, dtype=np.float32)
        n, ctx_len = context.shape
        P = self.patch_size
        if ctx_len % P != 0:
            raise ValueError(
                f"context length {ctx_len} not divisible by patch_size {P}"
            )
        if horizon % P != 0:
            raise ValueError(f"horizon {horizon} not divisible by patch_size {P}")
        context_patches = ctx_len // P
        pred_patches = horizon // P

        is_moiraic = self.model_name == "moiraic"
        Q = int(self.module.num_quantiles)

        # Zero-pad the prediction region (masked internally — see module docstring).
        full = np.concatenate(
            [context, np.zeros((n, horizon), dtype=np.float32)], axis=1
        )

        fq_buffer: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = full[i : i + batch_size]
            batch = make_batch(chunk, P, context_patches, pred_patches, self.device)
            with torch.no_grad():
                result = self.module(**batch, training_mode=False)
            result_np = result.detach().cpu().float().numpy()
            fq_buffer.append(
                _extract_fq(
                    result_np,
                    is_moiraic,
                    self.num_predict_token,
                    Q,
                    P,
                    context_patches,
                    pred_patches,
                )
            )
        return np.concatenate(fq_buffer, axis=0).astype(np.float32)


class MoiraicAdapter(_CustomMoiraiAdapter):
    """Custom causal/decoder model (``moiraic``)."""

    model_name = "moiraic"


class MoiraieAdapter(_CustomMoiraiAdapter):
    """Custom bidirectional/encoder model (``moiraie``)."""

    model_name = "moiraie"


class MoiraiXAdapter(ForecastAdapter):
    """Adapter for ``MoiraiXModule`` checkpoints.

    Loads via ``MoiraiXModule.from_pretrained`` so it works with any checkpoint
    trained through the unified moiraix path (decoder, encoder, encoder_ar, etc.).
    The ``predict_next`` flag on the loaded module selects the extraction path:
    True → causal/next-token (read from last context patch), False → encoder
    (read from prediction patches), matching ``_extract_fq``'s ``is_moiraic`` arg.
    """

    def __init__(self, ckpt: str, device: str = "cpu", **_ignored):
        from uni2ts.model.moiraix.module import MoiraiXModule

        self.device = device
        self.module = MoiraiXModule.from_pretrained(ckpt).eval().to(device)
        self.patch_size = int(self.module.patch_size)
        self.num_predict_token = int(self.module.num_predict_token)
        self.quantile_levels = tuple(
            getattr(self.module, "quantile_levels", DEFAULT_QUANTILE_LEVELS)
        )
        if len(self.quantile_levels) != int(self.module.num_quantiles):
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS

    def predict_quantiles(
        self,
        context: np.ndarray,
        horizon: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch

        from experiments.mech_interp.lib import make_batch

        context = np.asarray(context, dtype=np.float32)
        n, ctx_len = context.shape
        P = self.patch_size
        if ctx_len % P != 0:
            raise ValueError(
                f"context length {ctx_len} not divisible by patch_size {P}"
            )
        if horizon % P != 0:
            raise ValueError(f"horizon {horizon} not divisible by patch_size {P}")
        context_patches = ctx_len // P
        pred_patches = horizon // P

        Q = int(self.module.num_quantiles)
        is_predict_next = bool(self.module.predict_next)

        full = np.concatenate(
            [context, np.zeros((n, horizon), dtype=np.float32)], axis=1
        )

        fq_buffer: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = full[i : i + batch_size]
            batch = make_batch(chunk, P, context_patches, pred_patches, self.device)
            with torch.no_grad():
                result = self.module(**batch, training_mode=False)
            result_np = result.detach().cpu().float().numpy()
            fq_buffer.append(
                _extract_fq(
                    result_np,
                    is_predict_next,
                    self.num_predict_token,
                    Q,
                    P,
                    context_patches,
                    pred_patches,
                )
            )
        return np.concatenate(fq_buffer, axis=0).astype(np.float32)
