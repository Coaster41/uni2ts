"""Differentiable wrapper for MoiraiX and Moirai2 modules.

Both share the uni2ts packed batch contract and both return ``preds * scale + loc``
(denormalized, ``[B, S, npt*Q*P]``) when called with ``training_mode=False``. The
only difference is the readout slot, which is exactly the branch in
``block2_stress/models/custom_moirai.py::_extract_fq`` — ported here to torch so
the graph survives.

We call ``Module.forward`` directly, never ``*Forecast``: the inference wrappers
run under ``torch.no_grad``, AR-unroll, and post-process quantiles, all of which
either kill the graph or make it enormous.
"""
from __future__ import annotations

import torch

from .base import DEFAULT_QUANTILE_LEVELS, GradForecaster, make_batch_torch


class MoiraiGradForecaster(GradForecaster):
    """Differentiable wrapper over any uni2ts moirai-family module.

    ``kind`` selects the module class, and that choice is load-bearing:
    ``moiraic``/``moiraie`` checkpoints predate the moiraix flags and carry
    *no* ``causal``/``mask_inputs``/``predict_next`` in ``config.json``. Loading
    one through ``MoiraiXModule`` would silently fall back to the module defaults
    (``predict_next=False``) and read the forecast out of the wrong slot. The
    preset subclasses inject the correct flags, so use them.
    """

    def __init__(self, ckpt: str, kind: str = "moiraix", device: str = "cuda:0"):
        if kind == "moirai2":
            from uni2ts.model.moirai2 import Moirai2Module as Module
        elif kind == "moiraix":
            from uni2ts.model.moiraix.module import MoiraiXModule as Module
        elif kind == "moiraic":
            from uni2ts.model.moiraic.module import MoiraicModule as Module
        elif kind == "moiraie":
            from uni2ts.model.moiraie.module import MoiraieModule as Module
        else:
            raise ValueError(
                f"unknown kind {kind!r} (expected moiraix|moirai2|moiraic|moiraie)"
            )

        self.kind = kind
        self.device = device
        self.module = Module.from_pretrained(ckpt).to(device).eval()
        self.module.requires_grad_(False)  # freeze weights; we only want dL/dx

        self.patch_size = int(self.module.patch_size)
        self.npt = int(self.module.num_predict_token)
        self.num_quantiles = int(self.module.num_quantiles)
        levels = tuple(getattr(self.module, "quantile_levels", DEFAULT_QUANTILE_LEVELS))
        self.quantile_levels = (
            levels if len(levels) == self.num_quantiles else DEFAULT_QUANTILE_LEVELS
        )
        # moirai2 exposes no `predict_next`: it is causal / next-token by
        # construction (packed_causal_attention_mask + last-context-patch readout).
        self.predict_next = bool(getattr(self.module, "predict_next", True))

    def quantiles(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        n, ctx = context.shape
        Cp, Hp = self._check_geometry(ctx, horizon)
        P, Q, npt = self.patch_size, self.num_quantiles, self.npt

        # Zero-pad the prediction region. The encoder overwrites it with
        # mask_encoding and the causal model never attends to it, so its value
        # is irrelevant; `new_zeros` keeps the graph.
        full = torch.cat([context, context.new_zeros(n, horizon)], dim=1)
        batch = make_batch_torch(full, P, Cp, Hp)

        preds = self.module(**batch, training_mode=False)  # [n, S, npt*Q*P]

        if self.predict_next:
            # Causal: everything is emitted from the last context patch.
            blk = preds[:, Cp - 1, :].reshape(n, npt, Q, P)[:, :Hp]  # [n, Hp, Q, P]
        else:
            # Encoder: each prediction slot emits its own patch; take its first
            # next-token block.
            blk = preds[:, Cp : Cp + Hp, :].reshape(n, Hp, npt, Q, P)[:, :, 0]
        return blk.permute(0, 2, 1, 3).reshape(n, Q, Hp * P)  # [n, Q, H]
