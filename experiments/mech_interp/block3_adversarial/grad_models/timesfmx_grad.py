"""Differentiable wrapper for the TimesFMX fork's module.

``TimesFMXModule.forward`` returns ``(preds, target, loc, scale)`` with ``preds``
in *normalized* space, flat over ``o * num_heads_out`` where ``o = npt *
patch_size`` and ``num_heads_out = num_quantiles + predict_mean`` (a leading mean
column when ``predict_mean``). ``loc``/``scale`` are ``[B, S, 1]`` per-position.

This mirrors ``timesfm/pretrain/forecast.py::_run`` + ``read_block`` for the
first (and here only) AR step, minus the post-hoc quantile-crossing fix and
flip-invariance pass that ``TimesFMXForecast`` applies — the attack sees the raw
head. See HANDOFF §5c / §6.5.
"""
from __future__ import annotations

import torch

from .base import DEFAULT_QUANTILE_LEVELS, GradForecaster, make_batch_torch


class TimesFMXGradForecaster(GradForecaster):
    def __init__(self, ckpt: str, device: str = "cuda:0", **_ignored):
        from timesfm.pretrain import TimesFMXModule

        self._init_from_module(
            TimesFMXModule.from_pretrained(ckpt), device
        )

    def _init_from_module(self, module, device: str) -> None:
        self.device = device
        self.module = module.to(device).eval()
        self.module.requires_grad_(False)

        self.patch_size = int(self.module.patch_size)
        self.npt = int(self.module.num_predict_token)
        self.num_quantiles = int(self.module.num_quantiles)
        levels = tuple(getattr(self.module, "quantile_levels", DEFAULT_QUANTILE_LEVELS))
        self.quantile_levels = (
            levels if len(levels) == self.num_quantiles else DEFAULT_QUANTILE_LEVELS
        )
        self.predict_next = bool(self.module.predict_next)
        self.q0 = int(self.module.predict_mean)  # offset past the mean column
        self.o = int(self.module.o)  # npt * patch_size
        self.num_heads_out = int(self.module.num_heads_out)

    @property
    def single_pass_cap(self) -> int:
        return self.o

    def quantiles(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        n, ctx = context.shape
        Cp, Hp = self._check_geometry(ctx, horizon)

        full = torch.cat([context, context.new_zeros(n, horizon)], dim=1)
        b = make_batch_torch(full, self.patch_size, Cp, Hp)

        preds, _, loc, scale = self.module(
            b["target"],
            b["observed_mask"],
            b["sample_id"],
            b["time_id"],
            b["variate_id"],
            b["prediction_mask"],
        )  # preds: [n, S, o * num_heads_out], normalized
        preds = preds.reshape(n, -1, self.o, self.num_heads_out)

        # Causal reads from the last context patch; the encoder variant reads
        # from the first masked slot.
        src = Cp - 1 if self.predict_next else Cp
        blk = preds[:, src, :horizon, :]  # [n, H, num_heads_out]
        blk = blk * scale[:, src].unsqueeze(-1) + loc[:, src].unsqueeze(-1)
        q = blk[..., self.q0 : self.q0 + self.num_quantiles]  # [n, H, Q]
        return q.permute(0, 2, 1)  # [n, Q, H]


class TimesFM25GradForecaster(TimesFMXGradForecaster):
    """Official TimesFM-2.5 200M (`google/timesfm-2.5-200m-pytorch`) as a
    differentiable white-box model.

    The fork was built for checkpoint parity with 2.5 (identical submodule names),
    so the official safetensors load into ``TimesFMXModule`` with
    ``strict=True`` — 232/232 keys, no shape mismatches — and the resulting median
    forecast matches 2.5's own ``decode()`` to ~1e-6·σ_ctx. That makes the whole
    white-box machinery (gradient saliency, FGSM/PGD) available on a 200M frontier
    model, and TimesFM is one of the models the paper itself evaluates.

    Caveat (HANDOFF §3a, trap #5): this reads the raw ``output_projection_point``
    head. 2.5's inference wrapper additionally blends the continuous-quantile head
    and applies quantile-crossing / flip-invariance fixes. The **median column is
    identical** between the two paths, so every median-based metric we use (sMAE,
    RED_E, displacement) is unaffected; only the non-median quantiles differ, so a
    `WQL` computed here will not match block2's cached timesfm25 forecasts.
    """

    def __init__(self, device: str = "cuda:0", **_ignored):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from timesfm.pretrain import TimesFMXModule
        from timesfm.timesfm_2p5 import timesfm_2p5_torch

        cls = timesfm_2p5_torch.TimesFM_2p5_200M_torch
        weights = hf_hub_download(
            repo_id=cls.DEFAULT_REPO_ID, filename=cls.WEIGHTS_FILENAME
        )
        module = TimesFMXModule(
            patch_size=32,
            num_predict_token=4,
            d_model=1280,
            d_ff=1280,
            num_layers=20,
            num_heads=16,
            predict_mean=True,
            # 2.5 ships output_projection_quantiles; without this the strict load
            # fails on unexpected keys.
            use_continuous_quantile_head=True,
            causal=True,
            mask_inputs=False,
            predict_next=True,
            norm_mode="revin_causal",
        )
        module.load_state_dict(load_file(weights), strict=True)
        self._init_from_module(module, device)
